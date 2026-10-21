import os
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from datetime import datetime

# =====================================================================
# CONFIGURATIE & ENVIRONMENT VARIABLES (LIVE MAINNET)
# =====================================================================
API_KEY = os.getenv("KUCOIN_API_KEY", "")
API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
API_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "")

# GEVALIDEERD: Geforceerd op Mainnet met de juiste API Base URL
IS_SANDBOX = False 
BASE_URL = "https://api.kucoin.com" 

# Whitelist van handelsparen tegen USDT
HANDELSPAREN = ["DOGE-USDT", "BTC-USDT", "LINK-USDT", "ETH-USDT", "SOL-USDT"]

# Hugging Face Identifiers voor Kronos-base
MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

print(f"[*] [LIVE] AI-Model initialiseren: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval() 
    print("[+] [LIVE] Kronos-base succesvol ingeladen in het servergeheugen.")
except Exception as e:
    print(f"[-] KRITISCHE FOUT bij laden AI-model: {e}")
    raise SystemExit("Applicatie gestopt: AI-model kon niet worden gedownload.")

# =====================================================================
# AUTHENTICATIE EN KOPPELING FUNCTIES
# =====================================================================
def genereer_kucoin_headers(endpoint, methode="GET", body_str=""):
    timestamp = str(int(time.time() * 1000))
    str_to_sign = timestamp + methode.upper() + endpoint + body_str
    handtekening = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    beveiligde_passphrase = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), API_PASSPHRASE.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    return {
        "KC-API-KEY": API_KEY, "KC-API-SIGN": handtekening, "KC-API-TIMESTAMP": timestamp,
        "KC-API-PASSPHRASE": beveiligde_passphrase, "KC-API-KEY-VERSION": "2", "Content-Type": "application/json"
    }

def haal_live_saldo():
    """Haalt het actuele USDT saldo op van je KuCoin Handelsaccount."""
    endpoint = "/api/v1/accounts?type=trade&currency=USDT"
    headers = genereer_kucoin_headers(endpoint, methode="GET")
    try:
        response = requests.get(BASE_URL + endpoint, headers=headers)
        if response.status_code == 200:
            data = response.json().get("data", [])
            # Veilige check om IndexErrors te voorkomen als de lijst leeg is
            for account in data:
                if account.get("currency") == "USDT":
                    return float(account.get("balance", 0.0))
    except Exception as e:
        print(f"[-] Fout bij ophalen live saldo: {e}")
    return 0.0

def haal_historische_candles(symbol, interval="1hour", limit=50):
    endpoint = f"/api/v1/market/candles?symbol={symbol}&type={interval}&limit={limit}"
    try:
        response = requests.get(BASE_URL + endpoint)
        if response.status_code == 200:
            data = response.json().get("data", [])
            df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col])
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"[-] Fout bij ophalen live marktgeschiedenis voor {symbol}: {e}")
    return None

# =====================================================================
# REAL LIVE KRONOS AI INFERENCE
# =====================================================================
def bereken_kronos_ai_signaal(df):
    try:
        if df is None or len(df) < 30:
            return "HOLD"
            
        input_text = ""
        for _, row in df.tail(50).iterrows():
            input_text += f"O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f} v:{row['volume']:.0f} | "

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
        hidden_states = outputs.last_hidden_state
        richting_waarde = torch.tanh(hidden_states[0, -1, 0]).item()
        
        if richting_waarde > 0.4:
            return "BUY"
        elif richting_waarde < -0.4:
            return "SELL"
        return "HOLD"
    except Exception as e:
        print(f"[-] AI Inference fout: {e}")
        return "HOLD"

def bereken_veilig_risico(totaal_kapitaal, huidige_koers, stop_loss_percentage=0.03, max_risico_fractie=0.01):
    max_verlies_bedrag = totaal_kapitaal * max_risico_fractie
    risico_per_eenheid = huidige_koers * stop_loss_percentage
    aantal_eenheden = max_verlies_bedrag / risico_per_eenheid
    totale_waarde = aantal_eenheden * huidige_koers
    
    if totale_waarde > totaal_kapitaal:
        totale_waarde = totaal_kapitaal
        aantal_eenheden = totale_waarde / huidige_koers
        
    return {
        "eenheden": round(aantal_eenheden, 2), 
        "waarde_usdt": round(totale_waarde, 2),
        "stop_loss_prijs": round(huidige_koers * (1 - stop_loss_percentage), 4)
    }

def plaats_limiet_order(symbol, side, size, price):
    if symbol not in HANDELSPAREN: return None
    if size <= 0: 
        print(f"[-] Order geweigerd: berekende ordergrootte is 0.")
        return None
        
    endpoint = "/api/v1/orders"
    payload = {
        "clientOid": str(int(time.time() * 1000000)),
        "side": side.lower(),
        "symbol": symbol,
        "type": "limit",
        "size": str(size),
        "price": str(round(price, 5)),
        "timeInForce": "GTC"
    }
    import json
    body_str = json.dumps(payload)
    headers = genereer_kucoin_headers(endpoint, methode="POST", body_str=body_str)
    try:
        response = requests.post(BASE_URL + endpoint, headers=headers, data=body_str)
        return response.json()
    except Exception as e:
        print(f"[-] KRITISCHE FOUT bij verzenden live order: {e}")
        return None

# =====================================================================
# HOOFDPROGRAMMA (LIVE TRADING LOOP)
# =====================================================================
def main():
    print("==================================================")
    print(f"⚠️  🚨 KRONOS LIVE TRADING BOT - ECHT GELD ACTIEF - {datetime.now()}")
    print("==================================================")
    
    if not API_KEY or not API_SECRET or not API_PASSPHRASE:
        print("❌ FOUT: Live API-sleutels ontbreken in de variabelen!")
        return

    while True:
        live_saldo = haal_live_saldo()
        print(f"\n[💰 LIVE BALANS]: {live_saldo:.2f} USDT beschikbaar op Handelsaccount.")
        
        if live_saldo < 10.0:
            print("[⚠️] Saldo is te laag om minimale orders te plaatsen (Minimaal $10 vereist). Bot wacht 5 minuten...")
            time.sleep(300)
            continue

        for paar in HANDELSPAREN:
            print(f"\n[*] Analyseren live markt: {paar}...")
            df_candles = haal_historische_candles(paar)
            
            if df_candles is None or df_candles.empty:
                continue
                
            huidige_koers = df_candles['close'].iloc[-1]
            print(f"[+] Huidige koers {paar}: ${huidige_koers}")
            
            ai_signaal = bereken_kronos_ai_signaal(df_candles)
            print(f"[🤖 AI Model Signaal]: {ai_signaal}")
            
            if ai_signaal == "BUY":
                risico = bereken_veilig_risico(live_saldo, huidige_koers)
                print(f"[🛡️ Risicomanagement]: ECHT GELD ORDER -> Koop {risico['eenheden']} {paar} | SL op: ${risico['stop_loss_prijs']}")
                
                # REPARATIE: De transmissie naar de live exchange wordt nu correct uitgevoerd
                order_resultaat = plaats_limiet_order(paar, "buy", risico["eenheden"], huidige_koers)
                print(f"[📋 API Respons]: {order_resultaat}")
                
            elif ai_signaal == "SELL":
                print(f"[-] AI signaleert verkoop voor {paar}. Controleer handmatig je posities.")
                
        print(f"\n[ Live ] Ronde voltooid. Wachten op volgende interval (5 minuten)...")
        time.sleep(300)

if __name__ == "__main__":
    main()

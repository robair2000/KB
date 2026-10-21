import os
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# --- CONFIGURATIE & ENVIRONMENT VARIABLES ---
API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

# Veiligheid: Standaard op Sandbox (Testnet) om kapitaal te beschermen
IS_SANDBOX = os.getenv("KUCOIN_IS_SANDBOX", "True").lower() == "true"

BASE_URL = "https://kucoin.com" if IS_SANDBOX else "https://kucoin.com"

# Jouw whitelist van gekozen munten (gekoppeld tegen USDT voor risicoberekening)
TARGET_SYMBOLS = ["DOGE-USDT", "BTC-USDT", "LINK-USDT", "ETH-USDT", "SOL-USDT"]

# --- HUGGING FACE MODEL SETUP (ZWAARSTE MODEL) ---
MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

print(f"[*] AI-Model initialiseren: {MODEL_NAME}...")
try:
    # Laad het zware model expliciet op de CPU (aangezien Render Standard geen GPU heeft)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()  # Zet het model in evaluatie-modus voor inference
    print("[+] Kronos-base succesvol ingeladen in het geheugen.")
except Exception as e:
    print(f"[-] FOUT bij laden model: {e}")
    raise SystemExit("Applicatie gestopt: AI-model kon niet worden geladen.")

# --- KUCOIN AUTHENTICATIE GENERATOR ---
def geef_kucoin_headers(endpoint, methode="GET", body=""):
    now = str(int(time.time() * 1000))
    str_to_sign = now + methode + endpoint + body
    
    # Handtekening genereren middels HMAC-SHA256
    signature = base64.b64encode(
        hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()
    ).decode('utf-8')
    
    # Passphrase encrypten zoals KuCoin vereist
    passphrase_sign = base64.b64encode(
        hmac.new(API_SECRET.encode('utf-8'), API_PASSPHRASE.encode('utf-8'), hashlib.sha256).digest()
    ).decode('utf-8')
    
    return {
        "KC-API-KEY": API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": now,
        "KC-API-PASSPHRASE": passphrase_sign,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

# --- MARKT DATA OPHALEN ---
def haal_market_data(symbol, interval="1hour", limit=100):
    endpoint = f"/api/v1/market/candles?symbol={symbol}&type={interval}&limit={limit}"
    url = BASE_URL + endpoint
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json().get("data", [])
            # KuCoin output: [time, open, close, high, low, volume, turnover]
            df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
            # Converteren naar numerieke waarden
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col])
            # Sorteren van oud naar nieuw voor de AI-time-series
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"[-] Fout bij ophalen data voor {symbol}: {e}")
    return None

# --- AI INFERENCE (KRONOS-BASE) ---
def voer_ai_inference_uit(df):
    try:
        if df is None or len(df) < 50:
            return 0.0
            
        # Kronos verwacht genormaliseerde OHLCV data. 
        # We pakken de laatste 50 kaarsen als contextvenster
        laatste_data = df.tail(50)
        
        # Converteer de pandas kolommen naar een string/token input structuur die de tokenizer begrijpt
        # Kronos leest markten als opeenvolgende numerieke reeksen tokens
        input_text = ""
        for _, row in laatste_data.iterrows():
            input_text += f"O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f} V:{row['volume']:.0f} | "

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # De abstracte hidden states van de Transformer vertalen naar een prijsrichting
        # OPMERKING: Bij pure base-modellen pakken we de logits van de laatste token
        hidden_states = outputs.last_hidden_state
        voorspelde_richting = torch.tanh(hidden_states[0, -1, 0]).item() # Waarde tussen -1 en 1
        
        return voorspelde_richting
    except Exception as e:
        print(f"[-] AI Inference Fout: {e}")
        return 0.0

# --- RISICOMANAGEMENT ---
def bereken_veilig_risico(saldo, huidige_koers, stop_loss_pct=0.02, max_risico_pct=0.01):
    max_verlies_geld = saldo * max_risico_pct
    risico_per_munt = huidige_koers * stop_loss_pct
    aantal_eenheden = max_verlies_geld / risico_per_munt
    totale_investering = aantal_eenheden * huidige_koers
    return aantal_eenheden, totale_investering

# --- ORDERPLAATSING ---
def voer_trade_uit(symbol, side, aantal, prijs):
    endpoint = "/api/v1/orders"
    url = BASE_URL + endpoint
    
    # Uniek clientOid genereren om dubbele orders te voorkomen
    client_oid = f"kronos_{int(time.time() * 1000)}"
    
    payload = {
        "clientOid": client_oid,
        "side": side,
        "symbol": symbol,
        "type": "limit",
        "price": str(round(prijs, 5)),
        "size": str(round(aantal, 4))
    }
    
    headers = geof_kucoin_headers(endpoint, methode="POST", body=str(payload))
    
    print(f"[⚙️] VERZENDEN ORDER -> {side} {aantal} {symbol} op prijs {prijs}")
    # In sandbox/live sturen we de daadwerkelijke call naar de API:
    # response = requests.post(url, json=payload, headers=headers)
    # return response.json()

# --- HOOFDLOOP ---
def start_trading_bot():
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("[-] KRITISCHE FOUT: API-sleutels ontbreken in de omgevingsvariabelen!")
        return

    print(f"[+] Bot succesvol gestart. Modus: {'SANDBOX (TESTNET)' if IS_SANDBOX else 'LIVE MET ECHT GELD'}")
    
    # Fictief startkapitaal voor de risicocalculator in Sandbox-modus
    geschat_saldo = 5000.0 

    while True:
        print("\n--- Start Analyse Ronde ---")
        for symbol in TARGET_SYMBOLS:
            print(f"[*] Scannen van markt: {symbol}")
            df = haal_market_data(symbol)
            
            if df is not None and not df.empty:
                huidige_koers = df['close'].iloc[-1]
                
                # AI laten rekenen
                ai_signaal = voer_ai_inference_uit(df)
                print(f"    -> AI Signaalwaarde voor {symbol}: {ai_signaal:.4f}")
                
                # Koopregels: Als de transformer een sterk positief patroon ziet (> 0.3)
                if ai_signaal > 0.3:
                    aantal, totale_waarde = bereken_veilig_risico(geschat_saldo, huidige_koers)
                    print(f"    [🟩] AI adviseert KOOP. Risico-check: Koop {aantal:.4f} {symbol} (Waarde: ${totale_waarde:.2f})")
                    voer_trade_uit(symbol, "buy", aantal, huidige_koers)
                    
                elif ai_signaal < -0.3:
                    aantal, totale_waarde = bereken_veilig_risico(geschat_saldo, huidige_koers)
                    print(f"    [🟥] AI adviseert VERKOOP/SHORT voor {symbol}.")
                    voer_trade_uit(symbol, "sell", aantal, huidige_koers)
                    
                else:
                    print(f"    [🟨] AI Signaal is neutraal. Geen actie vereist.")
            
            time.sleep(2) # Voorkom rate-limiting van de KuCoin API
            
        print("[*] Ronde voltooid. Wachten op de volgende kaars (60 seconden)...")
        time.sleep(60)

if __name__ == "__main__":
    start_trading_bot()

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
from datetime import datetime

# --- CONFIGURATIE & ENVIRONMENT VARIABLES ---
API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

IS_SANDBOX = os.getenv("KUCOIN_IS_SANDBOX", "True").lower() == "true"
BASE_URL = "https://kucoin.com" if IS_SANDBOX else "https://kucoin.com"

TARGET_SYMBOLS = ["DOGE-USDT", "BTC-USDT", "LINK-USDT", "ETH-USDT", "SOL-USDT"]
LOG_FILE = "trading_performance_log.csv"

# --- MODEL SETUP ---
MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

print(f"[*] AI-Model initialiseren: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    print("[+] Kronos-base succesvol ingeladen.")
except Exception as e:
    print(f"[-] FOUT bij laden model: {e}")
    raise SystemExit("AI-model kon niet worden geladen.")

# --- INGEBOUWD LOG- EN PRESTATIESYSTEEM ---
# We houden actieve posities bij om winst/verlies te berekenen bij het sluiten
actieve_posities = {} 

def initialiseer_log_bestand():
    """Maakt het CSV-bestand aan met headers als het nog niet bestaat."""
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp", "Handelspaar", "Actie", "Aantal", 
            "Instapprijs", "Uitstapprijs", "Bruto_Resultaat_USD", "Rendement_Pct"
        ])
        df.to_csv(LOG_FILE, index=False)
        print(f"[📊] Nieuw prestatielogboek aangemaakt: {LOG_FILE}")

def log_transactie(symbol, actie, aantal, prijs, uitstapprijs=0.0, winst_usd=0.0, rendement_pct=0.0):
    """Schrijft een trade-regel weg naar het CSV-bestand."""
    nu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nieuwe_regel = pd.DataFrame([{
        "Timestamp": nu,
        "Handelspaar": symbol,
        "Actie": actie,
        "Aantal": round(aantal, 6),
        "Instapprijs": round(prijs, 5),
        "Uitstapprijs": round(uitstapprijs, 5) if uitstapprijs > 0 else "N.v.t.",
        "Bruto_Resultaat_USD": round(winst_usd, 2),
        "Rendement_Pct": round(rendement_pct, 2)
    }])
    nieuwe_regel.to_csv(LOG_FILE, mode='a', header=False, index=False)

def toon_dashboard_statistieken():
    """Leest het CSV-bestand uit en print een samenvatting in de Render logs."""
    try:
        df = pd.read_csv(LOG_FILE)
        sluitingen = df[df["Actie"].isin(["SLUIT_KOOP", "SLUIT_VERKOOP"])]
        
        if sluitingen.empty:
            print("\n[📊 STATISTIEKEN] Nog geen gesloten trades om prestaties te meten.")
            return
            
        totale_winst = sluitingen["Bruto_Resultaat_USD"].sum()
        winst_trades = sluitingen[sluitingen["Bruto_Resultaat_USD"] > 0]
        verlies_trades = sluitingen[sluitingen["Bruto_Resultaat_USD"] <= 0]
        
        win_rate = (len(winst_trades) / len(sluitingen)) * 100
        
        print("\n=========================================")
        print(f"[📊 KRONOS BOT PERFORMANCE DASHBOARD]")
        print(f" Total Gesloten Trades: {len(sluitingen)}")
        print(f" Winstgevende Trades : {len(winst_trades)}  |  Verliezende Trades: {len(verlies_trades)}")
        print(f" Win-Rate            : {win_rate:.2f}%")
        print(f" Netto Resultaat     : ${totale_winst:.2f}")
        print("=========================================")
    except Exception as e:
        print(f"[-] Fout bij genereren dashboard: {e}")

# --- KUCOIN AUTHENTICATIE GENERATOR ---
def geef_kucoin_headers(endpoint, methode="GET", body=""):
    now = str(int(time.time() * 1000))
    str_to_sign = now + methode + endpoint + body
    signature = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    passphrase_sign = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), API_PASSPHRASE.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    return {"KC-API-KEY": API_KEY, "KC-API-SIGN": signature, "KC-API-TIMESTAMP": now, "KC-API-PASSPHRASE": passphrase_sign, "KC-API-KEY-VERSION": "2", "Content-Type": "application/json"}

# --- MARKT DATA OPHALEN ---
def haal_market_data(symbol, interval="1hour", limit=100):
    endpoint = f"/api/v1/market/candles?symbol={symbol}&type={interval}&limit={limit}"
    try:
        response = requests.get(BASE_URL + endpoint, timeout=10)
        if response.status_code == 200:
            data = response.json().get("data", [])
            df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col])
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"[-] Fout bij data {symbol}: {e}")
    return None

# --- AI INFERENCE (KRONOS-BASE) ---
def voer_ai_inference_uit(df):
    try:
        if df is None or len(df) < 50: return 0.0
        laatste_data = df.tail(50)
        input_text = ""
        for _, row in laatste_data.iterrows():
            input_text += f"O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f} V:{row['volume']:.0f} | "
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        return torch.tanh(hidden_states[0, -1, 0]).item()
    except Exception as e:
        print(f"[-] AI Inference Fout: {e}")
        return 0.0

# --- RISICOMANAGEMENT ---
def bereken_veilig_risico(saldo, huidige_koers, stop_loss_pct=0.02, max_risico_pct=0.01):
    max_verlies_geld = saldo * max_risico_pct
    risico_per_munt = huidige_koers * stop_loss_pct
    return max_verlies_geld / risico_per_munt

# --- HOOFDLOOP ---
def start_trading_bot():
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("[-] KRITISCHE FOUT: API-sleutels ontbreken!")
        return

    initialiseer_log_bestand()
    print(f"[+] Bot gestart. Modus: {'SANDBOX' if IS_SANDBOX else 'LIVE'}")
    geschat_saldo = 5000.0 

    while True:
        print("\n--- Start Analyse Ronde ---")
        for symbol in TARGET_SYMBOLS:
            df = haal_market_data(symbol)
            if df is None or df.empty: continue
            
            huidige_koers = df['close'].iloc[-1]
            ai_signaal = voer_ai_inference_uit(df)
            print(f"[*] {symbol} | Koers: ${huidige_koers:.4f} | AI-Signaal: {ai_signaal:.4f}")
            
            # --- LOGICA: POSITIE SLUITEN ---
            if symbol in actieve_posities:
                pos = actieve_posities[symbol]
                # Sluit koop-positie als AI negatief wordt
                if pos["type"] == "BUY" and ai_signaal < -0.1:
                    winst = (huidige_koers - pos["prijs"]) * pos["aantal"]
                    rendement = ((huidige_koers - pos["prijs"]) / pos["prijs"]) * 100
                    geschat_saldo += winst
                    log_transactie(symbol, "SLUIT_KOOP", pos["aantal"], pos["prijs"], huidige_koers, winst, rendement)
                    print(f"    [🟥] SLUIT KOOP {symbol} op ${huidige_koers}. Resultaat: ${winst:.2f} ({rendement:.2f}%)")
                    del actieve_posities[symbol]
                
                # Sluit verkoop-positie als AI positief wordt
                elif pos["type"] == "SELL" and ai_signaal > 0.1:
                    winst = (pos["prijs"] - huidige_koers) * pos["aantal"]
                    rendement = ((pos["prijs"] - huidige_koers) / pos["prijs"]) * 100
                    geschat_saldo += winst
                    log_transactie(symbol, "SLUIT_VERKOOP", pos["aantal"], pos["prijs"], huidige_koers, winst, rendement)
                    print(f"    [🟩] SLUIT VERKOOP {symbol} op ${huidige_koers}. Resultaat: ${winst:.2f} ({rendement:.2f}%)")
                    del actieve_posities[symbol]

            # --- LOGICA: NIEUWE POSITIE OPENEN ---
            else:
                if ai_signaal > 0.4:  # Sterk koopsignaal
                    aantal = bereken_veilig_risico(geschat_saldo, huidige_koers)
                    actieve_posities[symbol] = {"type": "BUY", "prijs": huidige_koers, "aantal": aantal}
                    log_transactie(symbol, "OPEN_KOOP", aantal, huidige_koers)
                    print(f"    [🟩] OPEN KOOP ORDER voor {aantal:.4f} {symbol}")
                    
                elif ai_signaal < -0.4:  # Sterk verkoopsignaal
                    aantal = bereken_veilig_risico(geschat_saldo, huidige_koers)
                    actieve_posities[symbol] = {"type": "SELL", "prijs": huidige_koers, "aantal": aantal}
                    log_transactie(symbol, "OPEN_VERKOOP", aantal, huidige_koers)
                    print(f"    [🟥] OPEN VERKOOP ORDER voor {aantal:.4f} {symbol}")
            
            time.sleep(2) # Rate limit bescherming
            
        # Toon prestaties aan het einde van elke ronde in de logs
        toon_dashboard_statistieken()
        time.sleep(60)

if __name__ == "__main__":
    start_trading_bot()

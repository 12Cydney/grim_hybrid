//+------------------------------------------------------------------+
//| history_dump.mq5                                                 |
//|                                                                  |
//| MT5 SCRIPT (not an EA) - Exports historical OHLC to CSV files    |
//| that the Linux Grim Hybrid collector can read.                   |
//|                                                                  |
//| Usage:                                                           |
//|   1. Copy to MQL5/Scripts/ folder in MT5                        |
//|   2. Compile (F7 in MetaEditor)                                 |
//|   3. Drag onto ANY chart                                        |
//|   4. Set Inputs (symbols, years) and click OK                   |
//|   5. Wait for "DONE" alert                                      |
//|                                                                  |
//| Output files land in MQL5/Files/grim_hybrid_history/            |
//|   e.g. XAUUSD_H1.csv, XAUUSD_M15.csv, XAUUSD_M1.csv             |
//+------------------------------------------------------------------+
#property copyright "CYDNEYTRADES"
#property version   "1.00"
#property strict
#property description "Dumps MT5 history to CSV for Grim Hybrid trainer"
#property script_show_inputs

// -------------------------------------------------------------------
input string InpSymbols = "XAUUSD,NAS100,US30,EURUSD,GBPUSD,USDJPY,AUDUSD";
input int    InpYears   = 3;
input string InpOutputSubfolder = "grim_hybrid_history";
input bool   InpVerbose = true;

// -------------------------------------------------------------------
string g_outSubfolder;
int    g_totalBarsWritten = 0;

//+------------------------------------------------------------------+
void OnStart()
{
   g_outSubfolder = InpOutputSubfolder;
   datetime endTime   = TimeCurrent();
   datetime startTime = endTime - (datetime)(InpYears * 365 * 24 * 60 * 60);

   Print("=== Grim Hybrid history dump ===");
   Print("  Symbols:  ", InpSymbols);
   Print("  Years:    ", InpYears);
   Print("  Range:    ", TimeToString(startTime), "  ->  ", TimeToString(endTime));
   Print("  Output:   MQL5/Files/", g_outSubfolder, "/");

   string parts[];
   int n = StringSplit(InpSymbols, ',', parts);
   if(n <= 0)
   {
      Alert("No symbols provided");
      return;
   }

   int okCount = 0;
   int failCount = 0;

   for(int i = 0; i < n; i++)
   {
      string sym = TrimSpaces(parts[i]);
      if(sym == "") continue;

      Print("");
      Print("--- ", sym, " ---");

      // Make sure the broker knows this symbol
      if(!SymbolSelect(sym, true))
      {
         Print("  WARN: SymbolSelect failed - trying alt names for ", sym);
         string alt = ResolveBrokerSymbol(sym);
         if(alt == "")
         {
            Print("  SKIP: no broker match for ", sym);
            failCount++;
            continue;
         }
         Print("  RESOLVED: ", sym, " -> ", alt);
         sym = alt;
      }

      // Small delay so MT5 can request history from server
      Sleep(200);

      bool ok = true;
      ok = ok && DumpTimeframe(sym, PERIOD_H1,  "H1",  startTime, endTime);
      ok = ok && DumpTimeframe(sym, PERIOD_M15, "M15", startTime, endTime);
      ok = ok && DumpTimeframe(sym, PERIOD_M1,  "M1",  startTime, endTime);

      if(ok) okCount++;
      else   failCount++;
   }

   Print("");
   Print("=== DONE ===");
   Print("  Symbols exported OK: ", okCount);
   Print("  Symbols failed:      ", failCount);
   Print("  Total bars written:  ", g_totalBarsWritten);
   Alert("Grim Hybrid history dump complete. OK=", okCount, " Failed=", failCount);
}

//+------------------------------------------------------------------+
//| Dump one symbol/timeframe to CSV                                 |
//+------------------------------------------------------------------+
bool DumpTimeframe(string symbol, ENUM_TIMEFRAMES tf, string tfName,
                   datetime startTime, datetime endTime)
{
   MqlRates rates[];
   int copied = CopyRates(symbol, tf, startTime, endTime, rates);
   if(copied <= 0)
   {
      Print("  ", tfName, ": no bars (err=", GetLastError(), ")");
      return false;
   }

   // We want the CANONICAL symbol name in filenames so the collector
   // can match easily.  If the broker uses "XAUUSDm" we still write
   // "XAUUSD_H1.csv" - normalise here.
   string canonical = CanonicalName(symbol);
   string filename = g_outSubfolder + "/" + canonical + "_" + tfName + ".csv";

   int fh = FileOpen(filename, FILE_WRITE | FILE_ANSI | FILE_CSV, ",");
   if(fh == INVALID_HANDLE)
   {
      Print("  ", tfName, ": cannot open ", filename, " err=", GetLastError());
      return false;
   }

   // Header - keep column names identical to what the Python collector expects
   FileWrite(fh, "time", "open", "high", "low", "close", "tick_volume");

   for(int i = 0; i < copied; i++)
   {
      FileWrite(fh,
                (long)rates[i].time,
                DoubleToString(rates[i].open, 8),
                DoubleToString(rates[i].high, 8),
                DoubleToString(rates[i].low, 8),
                DoubleToString(rates[i].close, 8),
                (long)rates[i].tick_volume);
   }

   FileClose(fh);
   g_totalBarsWritten += copied;

   if(InpVerbose)
      Print("  ", tfName, ": ", copied, " bars -> ", filename);
   return true;
}

//+------------------------------------------------------------------+
//| Alt-name resolver (matches Grim Hybrid config aliases)           |
//+------------------------------------------------------------------+
string ResolveBrokerSymbol(string canonical)
{
   string tries[];
   string U = canonical; StringToUpper(U);

   if(U == "XAUUSD")  { StringSplit("XAUUSD,GOLD,XAUUSDm,XAUUSD.,XAUUSDc",         ',', tries); }
   else if(U == "NAS100") { StringSplit("NAS100,USTEC,NAS100m,US100,USTEC.,NAS100.,USTECH", ',', tries); }
   else if(U == "US30")   { StringSplit("US30,DJ30,US30m,DOW30,DJ30.,US30.,WS30",           ',', tries); }
   else if(U == "EURUSD") { StringSplit("EURUSD,EURUSDm,EURUSD.,EURUSDc,EURUSD_",           ',', tries); }
   else if(U == "GBPUSD") { StringSplit("GBPUSD,GBPUSDm,GBPUSD.,GBPUSDc,GBPUSD_",           ',', tries); }
   else if(U == "USDJPY") { StringSplit("USDJPY,USDJPYm,USDJPY.,USDJPYc,USDJPY_",           ',', tries); }
   else if(U == "AUDUSD") { StringSplit("AUDUSD,AUDUSDm,AUDUSD.,AUDUSDc,AUDUSD_",           ',', tries); }
   else return "";

   for(int i = 0; i < ArraySize(tries); i++)
   {
      string t = TrimSpaces(tries[i]);
      if(SymbolSelect(t, true)) return t;
   }
   return "";
}

//+------------------------------------------------------------------+
//| Canonical name (strip broker suffixes like m, ., c, _)           |
//+------------------------------------------------------------------+
string CanonicalName(string brokerSymbol)
{
   string s = brokerSymbol;
   string upper = s; StringToUpper(upper);

   if(StringFind(upper, "XAU") >= 0 || upper == "GOLD")   return "XAUUSD";
   if(StringFind(upper, "NAS") >= 0 || StringFind(upper, "USTEC") >= 0 || StringFind(upper, "US100") >= 0) return "NAS100";
   if(StringFind(upper, "US30") >= 0 || StringFind(upper, "DJ30") >= 0 || StringFind(upper, "DOW") >= 0 || StringFind(upper, "WS30") >= 0) return "US30";
   if(StringFind(upper, "EURUSD") >= 0) return "EURUSD";
   if(StringFind(upper, "GBPUSD") >= 0) return "GBPUSD";
   if(StringFind(upper, "USDJPY") >= 0) return "USDJPY";
   if(StringFind(upper, "AUDUSD") >= 0) return "AUDUSD";
   return brokerSymbol;
}

//+------------------------------------------------------------------+
string TrimSpaces(string s)
{
   StringTrimLeft(s);
   StringTrimRight(s);
   return s;
}

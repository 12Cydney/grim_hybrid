//+------------------------------------------------------------------+
//| GrimHybridEA_ZMQ.mq5 - v1.0 (Grim Hybrid framework client)       |
//| Execution-only. ZERO strategy logic. Talks TCP JSON to server.  |
//+------------------------------------------------------------------+
#property copyright "CYDNEYTRADES"
#property version   "1.00"
#property strict
#property description "Grim Hybrid EA - dumb executor for the Grim Hybrid framework"

#import "ws2_32.dll"
   int    WSAStartup(ushort version, uint& wsadata[]);
   int    WSACleanup();
   int    socket(int af, int type, int protocol);
   int    connect(int s, uchar& addr[], int addrlen);
   int    send(int s, uchar& buf[], int len, int flags);
   int    recv(int s, uchar& buf[], int len, int flags);
   int    closesocket(int s);
   ushort htons(ushort hostshort);
   uint   inet_addr(uchar& cp[]);
#import

#define AF_INET       2
#define SOCK_STREAM   1
#define IPPROTO_TCP   6
#define INADDR_NONE   0xFFFFFFFF

input string InpSymbol            = "";
input int    InpMagicNumber       = 987654;
input int    InpUpdateInterval    = 20;
input int    InpMaxSpread         = 1000;
input bool   InpUseTrailingStop   = true;
input string InpServerIP          = "127.0.0.1";
input int    InpServerPort        = 5555;

// Decisions come from the server -> the EA merely executes them.
// No confidence gate on the EA side (already applied by the server).

string   g_symbol;
datetime g_lastPing = 0;
int      g_socket   = -1;
int      g_reqId    = 0;

struct SignalData {
   string action;         // BUY | SELL | HOLD
   double confidence;
   double tp_probability;
   double sl_probability;
   double sl_price;       // absolute price (new in v1)
   double tp_price;       // absolute price (new in v1)
   double sl_distance;    // pips (legacy field, kept for panel)
   double setup_score;
   string setup_grade;
   string session;
   string method;
   string model_type;
   string hold_reason;
   string sl_kind;
   string tp_kind;
   double rr_planned;
   string volatility_regime;
   int    direction;
};
SignalData g_signal;

int OnInit()
{
   g_symbol = (InpSymbol == "") ? _Symbol : InpSymbol;
   ZeroMemory(g_signal);
   g_signal.action = "WAITING";

   uint wsa[128]; ArrayInitialize(wsa, 0);
   if(WSAStartup(0x0202, wsa) != 0) return(INIT_FAILED);

   PingServer();
   EventSetTimer(InpUpdateInterval);
   Print("GrimHybridEA v1.0 started | symbol=", g_symbol, " server=", InpServerIP, ":", InpServerPort);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_socket != -1) { closesocket(g_socket); g_socket = -1; }
   WSACleanup();
}

void OnTimer()
{
   if(TimeCurrent() - g_lastPing > 10) { PingServer(); g_lastPing = TimeCurrent(); }

   double spread = (double)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD);
   if(spread > InpMaxSpread) return;

   ProcessSignal();
   if(InpUseTrailingStop) ApplyTrailingStop();
}

//------------------------------------------------------------------
// Networking
//------------------------------------------------------------------
void PackAddr(uchar& addr[])
{
   ArrayResize(addr, 16);
   ArrayInitialize(addr, 0);
   addr[0] = (uchar)(AF_INET & 0xFF);
   addr[1] = (uchar)((AF_INET >> 8) & 0xFF);
   ushort port = htons((ushort)InpServerPort);
   addr[2] = (uchar)(port & 0xFF);
   addr[3] = (uchar)((port >> 8) & 0xFF);
   uchar ip_str[]; StringToCharArray(InpServerIP, ip_str, 0, WHOLE_ARRAY, CP_UTF8);
   uint ip = inet_addr(ip_str);
   if(ip == INADDR_NONE) { addr[4]=127; addr[5]=0; addr[6]=0; addr[7]=1; return; }
   addr[4] = (uchar)( ip        & 0xFF);
   addr[5] = (uchar)((ip >> 8)  & 0xFF);
   addr[6] = (uchar)((ip >> 16) & 0xFF);
   addr[7] = (uchar)((ip >> 24) & 0xFF);
}

bool Connect()
{
   if(g_socket != -1) { closesocket(g_socket); g_socket = -1; }
   g_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
   if(g_socket < 0) return false;
   uchar addr[]; PackAddr(addr);
   if(connect(g_socket, addr, 16) < 0) { closesocket(g_socket); g_socket = -1; return false; }
   return true;
}

string SendRequest(string body)
{
   if(!Connect()) return "";
   string framed = body + "\n";
   uchar buf[]; int len = StringToCharArray(framed, buf, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(send(g_socket, buf, len, 0) < 0) { closesocket(g_socket); g_socket=-1; return ""; }
   string accum = "";
   uchar rbuf[65536];
   for(int i=0; i<200; i++)
   {
      ArrayInitialize(rbuf, 0);
      int r = recv(g_socket, rbuf, ArraySize(rbuf), 0);
      if(r <= 0) break;
      accum += CharArrayToString(rbuf, 0, r, CP_UTF8);
      if(StringFind(accum, "\n") >= 0) break;
   }
   closesocket(g_socket); g_socket = -1;
   int nl = StringFind(accum, "\n");
   if(nl >= 0) accum = StringSubstr(accum, 0, nl);
   return accum;
}

void PingServer() { SendRequest("{\"type\":\"PING\"}"); }

//------------------------------------------------------------------
// JSON mini-helpers (same shape as legacy)
//------------------------------------------------------------------
int GetJSONInt(string j, string k)
{
   string s = "\"" + k + "\":"; int p = StringFind(j, s); if(p < 0) return 0; p += StringLen(s);
   string v = "";
   for(int i=p; i<StringLen(j); i++){ushort c=StringGetCharacter(j,i); if(c==','||c=='}'||c==' ') break; v+=CharToString((uchar)c);}
   return (int)StringToInteger(v);
}
double GetJSONDouble(string j, string k)
{
   string s = "\"" + k + "\":"; int p = StringFind(j, s); if(p < 0) return 0; p += StringLen(s);
   string v = "";
   for(int i=p; i<StringLen(j); i++){ushort c=StringGetCharacter(j,i); if(c==','||c=='}'||c==' ') break; v+=CharToString((uchar)c);}
   return StringToDouble(v);
}
string GetJSONString(string j, string k)
{
   string s = "\"" + k + "\":\""; int p = StringFind(j, s); if(p < 0) return ""; p += StringLen(s);
   string v = "";
   for(int i=p; i<StringLen(j); i++){ushort c=StringGetCharacter(j,i); if(c=='"') break; v+=CharToString((uchar)c);}
   return v;
}

//------------------------------------------------------------------
// Signal cycle
//------------------------------------------------------------------
string BuildRequestJSON()
{
   g_reqId++;
   string j = "{";
   j += "\"type\":\"PRICE_DATA_MTF\",";
   j += "\"request_id\":" + IntegerToString(g_reqId) + ",";
   j += "\"symbol\":\"" + g_symbol + "\",";
   j += "\"timestamp\":" + IntegerToString(TimeCurrent()) + ",";
   j += "\"spread\":" + IntegerToString((int)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD)) + ",";
   j += "\"data_1h\":"  + GetOHLCArray(PERIOD_H1, 100) + ",";
   j += "\"data_15m\":" + GetOHLCArray(PERIOD_M15, 100) + ",";
   j += "\"data_1m\":"  + GetOHLCArray(PERIOD_M1, 100);
   j += "}";
   return j;
}

string GetOHLCArray(ENUM_TIMEFRAMES tf, int count)
{
   MqlRates r[]; ArraySetAsSeries(r, true);
   int copied = CopyRates(g_symbol, tf, 0, count, r);
   if(copied <= 0) return "[]";
   string s = "[";
   for(int i = copied - 1; i >= 0; i--)
   {
      if(i < copied - 1) s += ",";
      s += "{\"time\":" + IntegerToString(r[i].time)
         + ",\"open\":"       + DoubleToString(r[i].open, 5)
         + ",\"high\":"       + DoubleToString(r[i].high, 5)
         + ",\"low\":"        + DoubleToString(r[i].low, 5)
         + ",\"close\":"      + DoubleToString(r[i].close, 5)
         + ",\"tick_volume\":"+ IntegerToString((long)r[i].tick_volume) + "}";
   }
   s += "]";
   return s;
}

void ProcessSignal()
{
   string resp = SendRequest(BuildRequestJSON());
   if(resp == "") return;

   g_signal.action            = GetJSONString(resp, "action");
   g_signal.confidence        = GetJSONDouble(resp, "confidence");
   g_signal.tp_probability    = GetJSONDouble(resp, "tp_probability");
   g_signal.sl_probability    = GetJSONDouble(resp, "sl_probability");
   g_signal.sl_price          = GetJSONDouble(resp, "sl_price");
   g_signal.tp_price          = GetJSONDouble(resp, "tp_price");
   g_signal.sl_distance       = GetJSONDouble(resp, "sl_distance");
   g_signal.setup_score       = GetJSONDouble(resp, "setup_score");
   g_signal.setup_grade       = GetJSONString(resp, "setup_grade");
   g_signal.session           = GetJSONString(resp, "session");
   g_signal.method            = GetJSONString(resp, "method");
   g_signal.model_type        = GetJSONString(resp, "model_type");
   g_signal.hold_reason       = GetJSONString(resp, "hold_reason");
   g_signal.sl_kind           = GetJSONString(resp, "sl_kind");
   g_signal.tp_kind           = GetJSONString(resp, "tp_kind");
   g_signal.rr_planned        = GetJSONDouble(resp, "rr_planned");
   g_signal.volatility_regime = GetJSONString(resp, "volatility_regime");
   g_signal.direction         = GetJSONInt(resp, "direction");

   if(g_signal.action == "BUY" || g_signal.action == "SELL")
   {
      if(!HasPosition())
      {
         ExecuteFromServerDecision();
      }
   }
}

bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      return true;
   }
   return false;
}

void ExecuteFromServerDecision()
{
   // Server already computed SL/TP absolute prices.  EA just executes.
   int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   ENUM_ORDER_TYPE ot = (g_signal.action == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double price = (ot == ORDER_TYPE_BUY)
                  ? SymbolInfoDouble(g_symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(g_symbol, SYMBOL_BID);

   double sl = NormalizeDouble(g_signal.sl_price, digits);
   double tp = NormalizeDouble(g_signal.tp_price, digits);
   if(sl <= 0 || tp <= 0)
   {
      Print("ERROR: server returned zero SL/TP - aborting trade");
      return;
   }

   double lots = CalculateLotSize(MathAbs(price - sl));

   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = g_symbol;
   req.volume    = lots;
   req.type      = ot;
   req.price     = price;
   req.sl        = sl;
   req.tp        = tp;
   req.deviation = 10;
   req.magic     = InpMagicNumber;
   req.comment   = "GrimHybrid_" + g_signal.setup_grade + "_" + DoubleToString(g_signal.confidence*100,0);

   if(OrderSend(req, res))
      Print("Trade sent: ", g_signal.action, " lots=", lots,
            " score=", DoubleToString(g_signal.setup_score, 2),
            " conf=", DoubleToString(g_signal.confidence*100, 1), "%",
            " sl_kind=", g_signal.sl_kind, " tp_kind=", g_signal.tp_kind);
   else
      Print("Trade failed. Error: ", GetLastError());
}

double CalculateLotSize(double slDistance)
{
   // Basic risk-per-trade sizing (server risk % lives in config, but we
   // don't parse it here; use a safe default of 1%).  The Python server
   // is authoritative for anything strategic; this is just execution.
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk      = balance * 1.0 / 100.0;
   double tickVal   = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);
   double minLot    = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   double maxLot    = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   double lotStep   = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);
   if(tickSize == 0 || tickVal == 0) return minLot;
   double lots = risk / ((slDistance / tickSize) * tickVal);
   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = MathFloor(lots / lotStep) * lotStep;
   return lots;
}

void ApplyTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      double open  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
      int    dig   = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
      ENUM_POSITION_TYPE pt = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      double newSL = 0;
      if(pt == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(g_symbol, SYMBOL_BID);
         if(bid - open >= 50 * point)
         {
            newSL = NormalizeDouble(bid - 20 * point, dig);
            if(newSL > sl) ModifyPositionSL(t, newSL);
         }
      }
      else if(pt == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
         if(open - ask >= 50 * point)
         {
            newSL = NormalizeDouble(ask + 20 * point, dig);
            if(newSL < sl || sl == 0) ModifyPositionSL(t, newSL);
         }
      }
   }
}

void ModifyPositionSL(ulong ticket, double newSL)
{
   MqlTradeRequest r = {}; MqlTradeResult res = {};
   r.action   = TRADE_ACTION_SLTP;
   r.position = ticket;
   r.sl       = newSL;
   r.tp       = PositionGetDouble(POSITION_TP);
   OrderSend(r, res);
}

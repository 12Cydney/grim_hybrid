//+------------------------------------------------------------------+
//|                     GrimHybridEA_ZMQ.mq5                        |
//|          Grim Hybrid ML - TCP Communication (v4.0)               |
//|                                                                  |
//|  v4.0 - Grim Hybrid framework client                             |
//|         + preserves ENTIRE v3.1 dashboard / panel / features    |
//|         + adds Setup Score, Grade, SL/TP kind, Vol regime rows  |
//|         + uses server-computed sl_price/tp_price when provided  |
//|                                                                  |
//|  Strategy logic lives in the Python StrategyEngine. This EA is  |
//|  an EXECUTION CLIENT + visual dashboard - it never decides      |
//|  entries/exits by itself.                                       |
//+------------------------------------------------------------------+
#property copyright "CYDNEYTRADES"
#property version   "4.00"
#property strict
#property description "cydney's learning machine (grim hybrid v1 client)"

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

#define AF_INET      2
#define SOCK_STREAM  1
#define IPPROTO_TCP  6
#define INADDR_NONE  0xFFFFFFFF

input string InpSymbol           = "";
input double InpRiskPercent      = 1.0;
input double InpRiskRewardRatio  = 2.0;
input int    InpMaxSpread        = 300;     // points - skip when spread is abnormally wide (tune per symbol)
input int    InpEntryCooldownMin = 15;      // minutes to wait after an entry before the next one
input bool   InpUseTrailingStop  = false;
input int    InpTrailingStart    = 50;
input int    InpTrailingStep     = 10;
input int    InpMagicNumber      = 123456;
input int    InpUpdateInterval   = 20;
input double InpMinConfidence    = 0.65;
input string InpServerIP         = "127.0.0.1";
input int    InpServerPort       = 5555;
input bool   InpUseServerSLTP    = true;    // NEW v4.0 - use server-computed SL/TP prices
input bool   InpUseServerRisk    = true;    // v4.1 - size with the server risk engine's risk_pct

input group  "Visual Settings"
input string InpTitleText        = "GRIM HYBRID EA";
input color  InpTitleColor       = C'0,0,0';
input bool   InpShowBackground   = false;

#define CLR_PANEL_BG   C'0,255,255'
#define CLR_BORDER     C'0,200,255'
#define CLR_HEADER_BG  C'0,0,0'
#define CLR_TEXT       C'255,255,255'
#define CLR_TEXT_DIM   C'180,180,200'
#define CLR_POSITIVE   C'100,255,100'
#define CLR_NEGATIVE   C'255,100,100'
#define CLR_WARNING    C'255,200,60'
#define CLR_BUY        C'60,200,255'
#define CLR_SELL       C'255,110,110'
#define CLR_NEUTRAL    C'180,180,180'
#define CLR_BAR_BG     C'20,20,30'
#define CLR_TITLE      C'0,220,255'
#define CLR_SUBTITLE   C'150,200,240'
#define CLR_DIVIDER    C'0,180,220'
#define CLR_GRADE_A    C'120,255,120'
#define CLR_GRADE_B    C'200,255,120'
#define CLR_GRADE_C    C'255,220,120'
#define CLR_GRADE_D    C'255,160,120'
#define CLR_GRADE_F    C'255,110,110'

string   g_symbol;
datetime g_lastUpdate = 0;
datetime g_lastPingTime = 0;
datetime g_lastPredictionTime = 0;
datetime g_lastEntryTime = 0;
int      g_requestId = 0;
string   g_panelPrefix = "SB_";
int      g_socket = -1;

// -------------------- data structs --------------------
struct SignalData {
   string   action;
   double   confidence;
   double   tp_probability;
   double   sl_probability;
   string   session;
   string   liquidity_type;
   string   method;
   string   model_type;
   string   hold_reason;
   double   sl_distance;
   int      direction;
   int      htf_bias;
   string   model_version;
   datetime timestamp;

   // ---- v4.0 additions (Grim Hybrid engine) ----
   double   setup_score;
   string   setup_grade;
   string   setup_description;
   string   sl_kind;
   string   tp_kind;
   double   sl_price;
   double   tp_price;
   double   rr_planned;
   string   volatility_regime;

   // ---- v4.1 risk engine ----
   double   risk_pct;
};
SignalData g_currentSignal;

struct ServerStatus {
   bool     connected;
   datetime lastPing;
   bool     modelLoaded;
   int      modelCount;
   int      featureCount;   // v4.0
   bool     hasFallback;    // v4.0
   string   modelVersion;
};
ServerStatus g_serverStatus;

struct TradeInfo {
   bool     hasPosition;
   double   entryPrice;
   double   slPrice;
   double   tpPrice;
   double   currentRMultiple;
   double   floatingPL;
   double   entryConfidence;
   datetime entryTime;
};
TradeInfo g_tradeInfo;

//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = (InpSymbol == "") ? _Symbol : InpSymbol;

   ZeroMemory(g_currentSignal);
   ZeroMemory(g_serverStatus);
   ZeroMemory(g_tradeInfo);

   g_currentSignal.action = "WAITING";
   g_currentSignal.session = "---";
   g_currentSignal.setup_grade = "-";
   g_currentSignal.volatility_regime = "---";
   g_serverStatus.modelVersion = "---";

   Print("Grim Hybrid EA v4.0 | Symbol=", g_symbol,
         " Risk=", InpRiskPercent, "% Server=", InpServerIP, ":", InpServerPort);

   if(!InitializeSocket())
   {
      Print("ERROR: WSAStartup failed");
      return(INIT_FAILED);
   }

   if(InpShowBackground) CreateBackground();
   CreateSimpleTitle();
   CreatePanel();

   PingServer();
   EventSetTimer(InpUpdateInterval);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   CleanupSocket();
   DeleteBackground();
   DeleteSimpleTitle();
   DeletePanel();
   Print("Grim Hybrid EA stopped. Reason: ", reason);
}

void OnTimer()
{
   if(TimeCurrent() - g_lastPingTime > 10)
   {
      PingServer();
      g_lastPingTime = TimeCurrent();
   }

   SendPositionState();   // v4.1 - feeds the server's daily caps / exposure guardrails

   double spread = (double)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD);
   if(spread > InpMaxSpread) { UpdatePanel(); return; }

   UpdateTradeInfo();
   ProcessSignal();

   if(InpUseTrailingStop && g_tradeInfo.hasPosition)
      ApplyTrailingStop();

   UpdatePanel();
}

void OnTick()
{
   // trailing stop runs on timer; nothing to do per-tick
}

//+------------------------------------------------------------------+
//| WinSock                                                          |
//+------------------------------------------------------------------+
bool InitializeSocket()
{
   uint wsadata[128];
   ArrayInitialize(wsadata, 0);
   return (WSAStartup(0x0202, wsadata) == 0);
}

void CleanupSocket()
{
   if(g_socket != -1) { closesocket(g_socket); g_socket = -1; }
   WSACleanup();
}

//+------------------------------------------------------------------+
//| Build sockaddr_in from InpServerIP using inet_addr()             |
//+------------------------------------------------------------------+
void PackServerAddress(uchar& addr_buf[])
{
   ArrayResize(addr_buf, 16);
   ArrayInitialize(addr_buf, 0);

   addr_buf[0] = (uchar)(AF_INET & 0xFF);
   addr_buf[1] = (uchar)((AF_INET >> 8) & 0xFF);

   ushort port_net = htons((ushort)InpServerPort);
   addr_buf[2] = (uchar)(port_net & 0xFF);
   addr_buf[3] = (uchar)((port_net >> 8) & 0xFF);

   uchar ip_str[];
   StringToCharArray(InpServerIP, ip_str, 0, WHOLE_ARRAY, CP_UTF8);
   uint ip_net = inet_addr(ip_str);

   if(ip_net == INADDR_NONE)
   {
      Print("ERROR: Invalid IP '", InpServerIP, "' -> using 127.0.0.1");
      addr_buf[4] = 127; addr_buf[5] = 0; addr_buf[6] = 0; addr_buf[7] = 1;
      return;
   }
   addr_buf[4] = (uchar)( ip_net        & 0xFF);
   addr_buf[5] = (uchar)((ip_net >> 8)  & 0xFF);
   addr_buf[6] = (uchar)((ip_net >> 16) & 0xFF);
   addr_buf[7] = (uchar)((ip_net >> 24) & 0xFF);
}

bool ConnectToServer()
{
   if(g_socket != -1) { closesocket(g_socket); g_socket = -1; }

   g_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
   if(g_socket < 0) return false;

   uchar addr_buf[];
   PackServerAddress(addr_buf);

   if(connect(g_socket, addr_buf, 16) < 0)
   {
      closesocket(g_socket); g_socket = -1;
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Send newline-terminated JSON, read until '\n'                    |
//+------------------------------------------------------------------+
string SendRequest(string json_request)
{
   if(!ConnectToServer()) return "";

   string framed = json_request + "\n";
   uchar send_buf[];
   int len = StringToCharArray(framed, send_buf, 0, WHOLE_ARRAY, CP_UTF8) - 1;

   if(send(g_socket, send_buf, len, 0) < 0)
   {
      closesocket(g_socket); g_socket = -1;
      return "";
   }

   string accumulated = "";
   uchar recv_buf[65536];
   for(int i = 0; i < 200; i++)
   {
      ArrayInitialize(recv_buf, 0);
      int received = recv(g_socket, recv_buf, ArraySize(recv_buf), 0);
      if(received <= 0) break;
      accumulated += CharArrayToString(recv_buf, 0, received, CP_UTF8);
      if(StringFind(accumulated, "\n") >= 0) break;
   }

   closesocket(g_socket); g_socket = -1;

   int nl = StringFind(accumulated, "\n");
   if(nl >= 0) accumulated = StringSubstr(accumulated, 0, nl);
   return accumulated;
}

//+------------------------------------------------------------------+
void PingServer()
{
   string response = SendRequest("{\"type\":\"PING\"}");
   if(response == "")
   {
      g_serverStatus.connected = false;
      return;
   }
   g_serverStatus.connected = true;
   g_serverStatus.lastPing = TimeCurrent();

   int count = GetJSONInt(response, "model_count");
   g_serverStatus.modelCount = count;
   g_serverStatus.modelLoaded = (count > 0);
   g_serverStatus.featureCount = GetJSONInt(response, "feature_count");   // v4.0
   g_serverStatus.hasFallback  = GetJSONBool(response, "has_fallback"); // v4.0 (JSON bool, unquoted)
}

//+------------------------------------------------------------------+
//| v4.1 POSITION_STATE: open trades + recent Grim deals -> server    |
//| Stateless & idempotent: the last 2 days of deals are re-sent each |
//| tick; the server de-dups by ticket and buckets by NY trading day. |
//+------------------------------------------------------------------+
void SendPositionState()
{
   datetime now = TimeCurrent();
   long gmtOffset = (long)(now - TimeGMT());

   int    openCount = 0;
   double floating  = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      openCount++;
      floating += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
   }

   string deals = "[";
   if(HistorySelect(now - 2 * 86400, now + 60))
   {
      // positions opened by this EA (closing deals from SL/TP are matched by position id)
      long ourPositions[];
      int  nPos = 0;
      int  total = HistoryDealsTotal();
      for(int i = 0; i < total; i++)
      {
         ulong t = HistoryDealGetTicket(i);
         if(t == 0) continue;
         if(HistoryDealGetString(t, DEAL_SYMBOL) != g_symbol) continue;
         if(HistoryDealGetInteger(t, DEAL_MAGIC) != InpMagicNumber) continue;
         ArrayResize(ourPositions, nPos + 1);
         ourPositions[nPos++] = HistoryDealGetInteger(t, DEAL_POSITION_ID);
      }

      bool first = true;
      for(int i = 0; i < total; i++)
      {
         ulong t = HistoryDealGetTicket(i);
         if(t == 0) continue;
         if(HistoryDealGetString(t, DEAL_SYMBOL) != g_symbol) continue;
         bool ours = (HistoryDealGetInteger(t, DEAL_MAGIC) == InpMagicNumber);
         if(!ours)
         {
            long pid = HistoryDealGetInteger(t, DEAL_POSITION_ID);
            for(int k = 0; k < nPos; k++) if(ourPositions[k] == pid) { ours = true; break; }
         }
         if(!ours) continue;

         long entry = HistoryDealGetInteger(t, DEAL_ENTRY);
         string e = "OTHER";
         if(entry == DEAL_ENTRY_IN) e = "IN";
         else if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY || entry == DEAL_ENTRY_INOUT) e = "OUT";
         double pnl = HistoryDealGetDouble(t, DEAL_PROFIT) + HistoryDealGetDouble(t, DEAL_COMMISSION)
                    + HistoryDealGetDouble(t, DEAL_SWAP);

         if(!first) deals += ",";
         first = false;
         deals += "{\"ticket\":" + IntegerToString((long)t)
                + ",\"time\":"   + IntegerToString((long)HistoryDealGetInteger(t, DEAL_TIME))
                + ",\"entry\":\"" + e + "\""
                + ",\"pnl\":"    + DoubleToString(pnl, 2) + "}";
      }
   }
   deals += "]";

   string json = "{";
   json += "\"type\":\"POSITION_STATE\",";
   json += "\"symbol\":\"" + g_symbol + "\",";
   json += "\"gmt_offset\":" + IntegerToString(gmtOffset) + ",";
   json += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
   json += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
   json += "\"open_trades\":" + IntegerToString(openCount) + ",";
   json += "\"floating_pnl\":" + DoubleToString(floating, 2) + ",";
   json += "\"deals\":" + deals;
   json += "}";
   SendRequest(json);
}

void UpdateTradeInfo()
{
   g_tradeInfo.hasPosition = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      g_tradeInfo.hasPosition = true;
      g_tradeInfo.entryPrice  = PositionGetDouble(POSITION_PRICE_OPEN);
      g_tradeInfo.slPrice     = PositionGetDouble(POSITION_SL);
      g_tradeInfo.tpPrice     = PositionGetDouble(POSITION_TP);
      g_tradeInfo.floatingPL  = PositionGetDouble(POSITION_PROFIT);
      g_tradeInfo.entryTime   = (datetime)PositionGetInteger(POSITION_TIME);

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double riskDistance = MathAbs(g_tradeInfo.entryPrice - g_tradeInfo.slPrice);
      if(riskDistance > 0)
      {
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(posType == POSITION_TYPE_BUY)
            g_tradeInfo.currentRMultiple = (currentPrice - g_tradeInfo.entryPrice) / riskDistance;
         else
            g_tradeInfo.currentRMultiple = (g_tradeInfo.entryPrice - currentPrice) / riskDistance;
      }
      break;
   }
}

void ProcessSignal()
{
   string response = SendRequest(BuildRequestJSON());
   Print("RAW RESPONSE: ", StringSubstr(response, 0, 300));
   if(response == "") return;

   ParseSignalResponse(response);
   g_lastPredictionTime = TimeCurrent();

   if(g_currentSignal.action == "BUY" || g_currentSignal.action == "SELL")
   {
      bool cooled = (TimeCurrent() - g_lastEntryTime) >= (InpEntryCooldownMin * 60);
      if(!g_tradeInfo.hasPosition && cooled && g_currentSignal.confidence >= InpMinConfidence)
      {
         ENUM_ORDER_TYPE orderType = (g_currentSignal.action == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
         ExecuteTrade(orderType, g_currentSignal.confidence, g_currentSignal.session);
         g_tradeInfo.entryConfidence = g_currentSignal.confidence;
      }
   }
}

void ParseSignalResponse(string json)
{
   g_currentSignal.action            = GetJSONString(json, "action");
   g_currentSignal.confidence        = GetJSONDouble(json, "confidence");
   g_currentSignal.tp_probability    = GetJSONDouble(json, "tp_probability");
   g_currentSignal.sl_probability    = GetJSONDouble(json, "sl_probability");
   g_currentSignal.session           = GetJSONString(json, "session");
   g_currentSignal.liquidity_type    = GetJSONString(json, "liquidity_type");
   g_currentSignal.method            = GetJSONString(json, "method");
   g_currentSignal.model_type        = GetJSONString(json, "model_type");
   g_currentSignal.hold_reason       = GetJSONString(json, "hold_reason");
   g_currentSignal.sl_distance       = GetJSONDouble(json, "sl_distance");
   g_currentSignal.direction         = GetJSONInt(json, "direction");
   g_currentSignal.htf_bias          = GetJSONInt(json, "htf_bias");
   g_currentSignal.model_version     = GetJSONString(json, "model_version");
   g_currentSignal.timestamp         = TimeCurrent();

   // ---- v4.0 additions ----
   g_currentSignal.setup_score       = GetJSONDouble(json, "setup_score");
   g_currentSignal.setup_grade       = GetJSONString(json, "setup_grade");
   g_currentSignal.setup_description = GetJSONString(json, "setup_description");
   g_currentSignal.sl_kind           = GetJSONString(json, "sl_kind");
   g_currentSignal.tp_kind           = GetJSONString(json, "tp_kind");
   g_currentSignal.sl_price          = GetJSONDouble(json, "sl_price");
   g_currentSignal.tp_price          = GetJSONDouble(json, "tp_price");
   g_currentSignal.rr_planned        = GetJSONDouble(json, "rr_planned");
   g_currentSignal.volatility_regime = GetJSONString(json, "volatility_regime");
   g_currentSignal.risk_pct          = GetJSONDouble(json, "risk_pct");   // v4.1

   g_serverStatus.modelVersion = g_currentSignal.model_version;
}

string BuildRequestJSON()
{
   g_requestId++;
   string json = "{";
   json += "\"type\":\"PRICE_DATA_MTF\",";
   json += "\"request_id\":" + IntegerToString(g_requestId) + ",";
   json += "\"symbol\":\"" + g_symbol + "\",";
   json += "\"timestamp\":" + IntegerToString(TimeCurrent()) + ",";
   json += "\"gmt_offset\":" + IntegerToString((long)(TimeCurrent() - TimeGMT())) + ",";   // v4.2 broker clock -> UTC
   json += "\"spread\":" + IntegerToString((int)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD)) + ",";
   json += "\"data_1h\":"  + GetOHLCArray(PERIOD_H1, 100) + ",";
   json += "\"data_15m\":" + GetOHLCArray(PERIOD_M15, 100) + ",";
   json += "\"data_1m\":"  + GetOHLCArray(PERIOD_M1, 100);
   json += "}";
   return json;
}

string GetOHLCArray(ENUM_TIMEFRAMES tf, int count)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   // start=1 -> CLOSED bars only. Index 0 is the still-forming bar, whose
   // OHLC the collector/trainer never sees -> feature drift. (Grim fix #1)
   int copied = CopyRates(g_symbol, tf, 1, count, rates);
   if(copied <= 0) return "[]";

   string json = "[";
   for(int i = copied - 1; i >= 0; i--)
   {
      if(i < copied - 1) json += ",";
      json += "{";
      json += "\"time\":"        + IntegerToString(rates[i].time) + ",";
      json += "\"open\":"        + DoubleToString(rates[i].open, 5) + ",";
      json += "\"high\":"        + DoubleToString(rates[i].high, 5) + ",";
      json += "\"low\":"         + DoubleToString(rates[i].low, 5) + ",";
      json += "\"close\":"       + DoubleToString(rates[i].close, 5) + ",";
      json += "\"tick_volume\":" + IntegerToString((long)rates[i].tick_volume);
      json += "}";
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| JSON helpers                                                     |
//+------------------------------------------------------------------+
int GetJSONInt(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos += StringLen(search);
   string value = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      ushort c = StringGetCharacter(json, i);
      if(c == ',' || c == '}' || c == ' ') break;
      value += CharToString((uchar)c);
   }
   return (int)StringToInteger(value);
}

double GetJSONDouble(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos += StringLen(search);
   string value = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      ushort c = StringGetCharacter(json, i);
      if(c == ',' || c == '}' || c == ' ') break;
      value += CharToString((uchar)c);
   }
   return StringToDouble(value);
}

// JSON booleans are unquoted (true/false) - GetJSONString can't read them.
bool GetJSONBool(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return false;
   pos += StringLen(search);
   while(pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
   return (StringSubstr(json, pos, 4) == "true");
}

string GetJSONString(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos += StringLen(search);
   string value = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      ushort c = StringGetCharacter(json, i);
      if(c == '"') break;
      value += CharToString((uchar)c);
   }
   return value;
}

//+------------------------------------------------------------------+
//| Trading                                                          |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE orderType, double confidence, string session)
{
   double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   double price, sl, tp, slDistance;

   if(orderType == ORDER_TYPE_BUY)
      price = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
   else
      price = SymbolInfoDouble(g_symbol, SYMBOL_BID);

   // v4.0: prefer server-computed SL/TP (intelligent multi-location eval).
   // Falls back to legacy CalculateSLDistance() if server didn't provide them
   // or InpUseServerSLTP is disabled.
   if(InpUseServerSLTP && g_currentSignal.sl_price > 0 && g_currentSignal.tp_price > 0)
   {
      sl = NormalizeDouble(g_currentSignal.sl_price, digits);
      tp = NormalizeDouble(g_currentSignal.tp_price, digits);
      slDistance = MathAbs(price - sl);
      Print("Using SERVER SL/TP - kind=", g_currentSignal.sl_kind, "/", g_currentSignal.tp_kind,
            " R:R=", DoubleToString(g_currentSignal.rr_planned, 2));
   }
   else
   {
      slDistance = CalculateSLDistance();
      if(slDistance == 0) slDistance = 500 * point;
      double tpDistance = slDistance * InpRiskRewardRatio;
      if(orderType == ORDER_TYPE_BUY)
      {
         sl = NormalizeDouble(price - slDistance, digits);
         tp = NormalizeDouble(price + tpDistance, digits);
      }
      else
      {
         sl = NormalizeDouble(price + slDistance, digits);
         tp = NormalizeDouble(price - tpDistance, digits);
      }
   }

   double lots = CalculateLotSize(slDistance);

   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = g_symbol;
   request.volume    = lots;
   request.type      = orderType;
   request.price     = price;
   request.sl        = sl;
   request.tp        = tp;
   request.deviation = 10;
   request.magic     = InpMagicNumber;
   request.comment   = "GrimHybrid_" + g_currentSignal.setup_grade + "_"
                      + session + "_" + DoubleToString(confidence * 100, 0);

   if(OrderSend(request, result))
   {
      g_lastEntryTime = TimeCurrent();
      Print("Trade: ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"),
            " | Lots: ", lots, " | Conf: ", DoubleToString(confidence * 100, 1), "%",
            " | Grade: ", g_currentSignal.setup_grade);
   }
   else
      Print("Trade failed. Error: ", GetLastError());
}

double CalculateSLDistance()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(g_symbol, PERIOD_M15, 0, 20, rates) <= 0) return 0;

   double highest = rates[0].high, lowest = rates[0].low;
   for(int i = 1; i < 20; i++)
   {
      if(rates[i].high > highest) highest = rates[i].high;
      if(rates[i].low  < lowest)  lowest  = rates[i].low;
   }
   return (highest - lowest) * 0.5;
}

double CalculateLotSize(double slDistance)
{
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskPct   = (InpUseServerRisk && g_currentSignal.risk_pct > 0) ? g_currentSignal.risk_pct : InpRiskPercent;
   double risk      = balance * riskPct / 100.0;
   double tickVal   = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize == 0 || tickVal == 0) return SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);

   double slPoints = slDistance / tickSize;
   double lots     = risk / (slPoints * tickVal);

   double minLot  = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(minLot, lots);
   lots = MathMin(maxLot, lots);
   lots = MathFloor(lots / lotStep) * lotStep;
   return lots;
}

void ApplyTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double point     = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
      int    digits    = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double newSL = 0;

      if(posType == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(g_symbol, SYMBOL_BID);
         if(bid - openPrice >= InpTrailingStart * point)
         {
            newSL = NormalizeDouble(bid - InpTrailingStep * point, digits);
            if(newSL > currentSL) ModifyPosition(ticket, newSL);
         }
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
         if(openPrice - ask >= InpTrailingStart * point)
         {
            newSL = NormalizeDouble(ask + InpTrailingStep * point, digits);
            if(newSL < currentSL || currentSL == 0) ModifyPosition(ticket, newSL);
         }
      }
   }
}

void ModifyPosition(ulong ticket, double newSL)
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl       = newSL;
   request.tp       = PositionGetDouble(POSITION_TP);
   if(!OrderSend(request, result))
      Print("Failed to modify SL. Error: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Panel                                                            |
//+------------------------------------------------------------------+
void CreateBackground()
{
   int chartWidth   = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int chartHeight  = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   int panelWidth   = 300;
   int availableWidth  = chartWidth - panelWidth - 40;
   int availableHeight = chartHeight - 100;
   int imageWidth   = (int)(availableWidth * 0.85);
   int imageHeight  = (int)(imageWidth * 0.59);
   int imageX       = panelWidth + (availableWidth - imageWidth) / 2;
   int imageY       = 80 + ((chartHeight - 80) - imageHeight) / 2;

   if(ObjectCreate(0, g_panelPrefix + "BgImage", OBJ_BITMAP_LABEL, 0, 0, 0))
   {
      ObjectSetString (0, g_panelPrefix + "BgImage", OBJPROP_BMPFILE,    "\\Files\\background.bmp");
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_XDISTANCE,  imageX);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_YDISTANCE,  imageY);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_XSIZE,      imageWidth);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_YSIZE,      imageHeight);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_CORNER,     CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_BACK,       true);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, g_panelPrefix + "BgImage", OBJPROP_HIDDEN,     true);
   }
}

void DeleteBackground() { ObjectDelete(0, g_panelPrefix + "BgImage"); }

void CreateSimpleTitle()
{
   int chartWidth = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int panelWidth = 300;
   int titleX     = panelWidth + ((chartWidth - panelWidth) / 2);

   ObjectCreate(0, g_panelPrefix + "Title", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_XDISTANCE,  titleX);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_YDISTANCE,  20);
   ObjectSetString (0, g_panelPrefix + "Title", OBJPROP_TEXT,       InpTitleText);
   ObjectSetString (0, g_panelPrefix + "Title", OBJPROP_FONT,       "Arial");
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_FONTSIZE,   20);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_COLOR,      InpTitleColor);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_ANCHOR,     ANCHOR_CENTER);
   ObjectSetInteger(0, g_panelPrefix + "Title", OBJPROP_SELECTABLE, false);

   ObjectCreate(0, g_panelPrefix + "Version", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, g_panelPrefix + "Version", OBJPROP_XDISTANCE,  chartWidth - 70);
   ObjectSetInteger(0, g_panelPrefix + "Version", OBJPROP_YDISTANCE,  15);
   ObjectSetString (0, g_panelPrefix + "Version", OBJPROP_TEXT,       "v4.0");
   ObjectSetString (0, g_panelPrefix + "Version", OBJPROP_FONT,       "Arial");
   ObjectSetInteger(0, g_panelPrefix + "Version", OBJPROP_FONTSIZE,   9);
   ObjectSetInteger(0, g_panelPrefix + "Version", OBJPROP_COLOR,      C'150,150,150');
   ObjectSetInteger(0, g_panelPrefix + "Version", OBJPROP_SELECTABLE, false);
}

void DeleteSimpleTitle()
{
   ObjectDelete(0, g_panelPrefix + "Title");
   ObjectDelete(0, g_panelPrefix + "Version");
}

void CreatePanel()
{
   int x = 30, y = 50, panelWidth = 260;

   CreateLabel(g_panelPrefix + "PanelTitle", x, y, "GRIM HYBRID ML", CLR_TITLE, 10, true); y += 22;

   // -------------- SYSTEM STATUS --------------
   CreateLabel(g_panelPrefix + "Sec1", x, y, "SYSTEM STATUS", CLR_TEXT_DIM, 8, true); y += 18;
   CreateLabel(g_panelPrefix + "ServerLbl",    x, y, "Server",     CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "ServerVal",    x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "HeartbeatLbl", x, y, "Heartbeat",  CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "HeartbeatVal", x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "ModelLbl",     x, y, "Model",      CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "ModelVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "FeatLbl",      x, y, "Features",   CLR_TEXT_DIM, 9, false);  // v4.0
   CreateLabel(g_panelPrefix + "FeatVal",      x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "VersionLbl",   x, y, "Version",    CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "VersionVal",   x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "LastPredLbl",  x, y, "Last Signal",CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "LastPredVal",  x + 95, y, "---",   CLR_TEXT, 9, false); y += 22;

   CreateRectangle(g_panelPrefix + "Div1", x, y, panelWidth - 20, 1, CLR_DIVIDER, CLR_DIVIDER, 0); y += 8;

   // -------------- CURRENT SIGNAL --------------
   CreateLabel(g_panelPrefix + "Sec2", x, y, "CURRENT SIGNAL", CLR_TEXT_DIM, 8, true); y += 18;
   CreateLabel(g_panelPrefix + "SymbolLbl",    x, y, "Symbol",   CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "SymbolVal",    x + 95, y, "---", CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "SessionLbl",   x, y, "Session",  CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "SessionVal",   x + 95, y, "---", CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "DirectionLbl", x, y, "Signal",   CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "DirectionVal", x + 95, y, "---", CLR_TEXT, 10, true); y += 18;

   // Setup grade & score - v4.0 additions
   CreateLabel(g_panelPrefix + "GradeLbl",  x, y, "Grade",  CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "GradeVal",  x + 95, y, "-",  CLR_TEXT, 11, true); y += 16;
   CreateLabel(g_panelPrefix + "ScoreLbl",  x, y, "Score",  CLR_TEXT_DIM, 9, false);
   CreateRectangle(g_panelPrefix + "ScoreBarBG", x + 60, y + 2, 140, 10, CLR_BAR_BG, CLR_BAR_BG, 0);
   CreateRectangle(g_panelPrefix + "ScoreBar",   x + 60, y + 2, 1,   10, CLR_TITLE,  CLR_TITLE, 0);
   CreateLabel(g_panelPrefix + "ScoreVal",  x + 205, y, "0.00", CLR_TITLE, 9, false); y += 18;

   // TP / SL probability bars
   CreateLabel(g_panelPrefix + "TPLbl", x, y, "TP", CLR_POSITIVE, 9, false);
   CreateRectangle(g_panelPrefix + "TPBarBG", x + 30, y + 2, 140, 10, CLR_BAR_BG, CLR_BAR_BG, 0);
   CreateRectangle(g_panelPrefix + "TPBar",   x + 30, y + 2, 1,   10, CLR_POSITIVE, CLR_POSITIVE, 0);
   CreateLabel(g_panelPrefix + "TPVal", x + 180, y, "0%", CLR_POSITIVE, 9, false); y += 16;
   CreateLabel(g_panelPrefix + "SLLbl", x, y, "SL", CLR_NEGATIVE, 9, false);
   CreateRectangle(g_panelPrefix + "SLBarBG", x + 30, y + 2, 140, 10, CLR_BAR_BG, CLR_BAR_BG, 0);
   CreateRectangle(g_panelPrefix + "SLBar",   x + 30, y + 2, 1,   10, CLR_NEGATIVE, CLR_NEGATIVE, 0);
   CreateLabel(g_panelPrefix + "SLVal", x + 180, y, "0%", CLR_NEGATIVE, 9, false); y += 18;

   CreateLabel(g_panelPrefix + "ConfidenceLbl", x, y, "Confidence", CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "ConfidenceVal", x + 95, y, "---",   CLR_TEXT, 9, true); y += 15;
   CreateLabel(g_panelPrefix + "RRLbl",         x, y, "R:R",        CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "RRVal",         x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "SLKindLbl",     x, y, "SL Type",    CLR_TEXT_DIM, 9, false);  // v4.0
   CreateLabel(g_panelPrefix + "SLKindVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "TPKindLbl",     x, y, "TP Type",    CLR_TEXT_DIM, 9, false);  // v4.0
   CreateLabel(g_panelPrefix + "TPKindVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "VolRegLbl",     x, y, "Volatility", CLR_TEXT_DIM, 9, false);  // v4.0
   CreateLabel(g_panelPrefix + "VolRegVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "SLDistLbl",     x, y, "SL Dist",    CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "SLDistVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "MethodLbl",     x, y, "Method",     CLR_TEXT_DIM, 9, false);  // v4.0
   CreateLabel(g_panelPrefix + "MethodVal",     x + 95, y, "---",   CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "HoldLbl",       x, y, "Reason",     CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "HoldVal",       x + 95, y, "---",   CLR_WARNING, 8, false); y += 20;

   CreateRectangle(g_panelPrefix + "Div2", x, y, panelWidth - 20, 1, CLR_DIVIDER, CLR_DIVIDER, 0); y += 8;

   // -------------- TRADE INFO --------------
   CreateLabel(g_panelPrefix + "Sec3", x, y, "TRADE INFO", CLR_TEXT_DIM, 8, true); y += 18;
   CreateLabel(g_panelPrefix + "StatusLbl",    x, y, "Status",      CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "StatusVal",    x + 95, y, "NO POSITION", CLR_TEXT_DIM, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "EntryLbl",     x, y, "Entry",       CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "EntryVal",     x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "TradeSLLbl",   x, y, "SL",          CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "TradeSLVal",   x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "TradeTPLbl",   x, y, "TP",          CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "TradeTPVal",   x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "RMultLbl",     x, y, "Current R",   CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "RMultVal",     x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "PLLbl",        x, y, "P/L",         CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "PLVal",        x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "EntryConfLbl", x, y, "Entry Conf",  CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "EntryConfVal", x + 95, y, "---",    CLR_TEXT, 9, false); y += 15;
   CreateLabel(g_panelPrefix + "TimeLbl",      x, y, "Duration",    CLR_TEXT_DIM, 9, false);
   CreateLabel(g_panelPrefix + "TimeVal",      x + 95, y, "---",    CLR_TEXT, 9, false);

   ChartRedraw();
}

void DeletePanel() { ObjectsDeleteAll(0, g_panelPrefix); }

void CreateLabel(string name, int x, int y, string text, color clr, int fontSize, bool bold)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE,  x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE,  y);
   ObjectSetString (0, name, OBJPROP_TEXT,       text);
   ObjectSetString (0, name, OBJPROP_FONT,       bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,   fontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,     ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void CreateRectangle(string name, int x, int y, int width, int height,
                     color bgColor, color borderColor, int borderWidth)
{
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE,   x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE,   y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE,       width);
   ObjectSetInteger(0, name, OBJPROP_YSIZE,       height);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR,     bgColor);
   ObjectSetInteger(0, name, OBJPROP_COLOR,       borderColor);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,       borderWidth);
   ObjectSetInteger(0, name, OBJPROP_CORNER,      CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE,  false);
}

color GradeColor(string grade)
{
   if(grade == "A+" || grade == "A") return CLR_GRADE_A;
   if(grade == "B")                  return CLR_GRADE_B;
   if(grade == "C")                  return CLR_GRADE_C;
   if(grade == "D")                  return CLR_GRADE_D;
   return CLR_GRADE_F;
}

color VolRegimeColor(string regime)
{
   if(regime == "NORMAL")  return CLR_POSITIVE;
   if(regime == "HIGH")    return CLR_WARNING;
   if(regime == "EXTREME") return CLR_NEGATIVE;
   if(regime == "LOW")     return CLR_TEXT_DIM;
   return CLR_NEUTRAL;
}

void UpdatePanel()
{
   // -------- SYSTEM STATUS --------
   ObjectSetString (0, g_panelPrefix + "ServerVal", OBJPROP_TEXT,
                    g_serverStatus.connected ? "RUNNING" : "OFFLINE");
   ObjectSetInteger(0, g_panelPrefix + "ServerVal", OBJPROP_COLOR,
                    g_serverStatus.connected ? CLR_POSITIVE : CLR_NEGATIVE);

   if(g_serverStatus.lastPing > 0)
   {
      int secAgo = (int)(TimeCurrent() - g_serverStatus.lastPing);
      ObjectSetString (0, g_panelPrefix + "HeartbeatVal", OBJPROP_TEXT, IntegerToString(secAgo) + "s ago");
      ObjectSetInteger(0, g_panelPrefix + "HeartbeatVal", OBJPROP_COLOR, secAgo > 30 ? CLR_WARNING : CLR_TEXT);
   }

   string modelTxt = g_serverStatus.modelLoaded
                     ? ("Loaded (" + IntegerToString(g_serverStatus.modelCount) + ")")
                     : "Rules Only";
   ObjectSetString (0, g_panelPrefix + "ModelVal", OBJPROP_TEXT, modelTxt);
   ObjectSetInteger(0, g_panelPrefix + "ModelVal", OBJPROP_COLOR,
                    g_serverStatus.modelLoaded ? CLR_POSITIVE : CLR_WARNING);

   ObjectSetString(0, g_panelPrefix + "FeatVal", OBJPROP_TEXT,
                   IntegerToString(g_serverStatus.featureCount));   // v4.0
   ObjectSetString(0, g_panelPrefix + "VersionVal", OBJPROP_TEXT, g_serverStatus.modelVersion);

   if(g_lastPredictionTime > 0)
   {
      int secAgo = (int)(TimeCurrent() - g_lastPredictionTime);
      ObjectSetString(0, g_panelPrefix + "LastPredVal", OBJPROP_TEXT, IntegerToString(secAgo) + "s ago");
   }

   // -------- CURRENT SIGNAL --------
   ObjectSetString(0, g_panelPrefix + "SymbolVal",  OBJPROP_TEXT, g_symbol);
   ObjectSetString(0, g_panelPrefix + "SessionVal", OBJPROP_TEXT, g_currentSignal.session);

   ObjectSetString(0, g_panelPrefix + "DirectionVal", OBJPROP_TEXT, g_currentSignal.action);
   color dirColor = CLR_NEUTRAL;
   if(g_currentSignal.action == "BUY")  dirColor = CLR_BUY;
   if(g_currentSignal.action == "SELL") dirColor = CLR_SELL;
   ObjectSetInteger(0, g_panelPrefix + "DirectionVal", OBJPROP_COLOR, dirColor);

   // Grade + score
   string grade = g_currentSignal.setup_grade == "" ? "-" : g_currentSignal.setup_grade;
   ObjectSetString (0, g_panelPrefix + "GradeVal", OBJPROP_TEXT,  grade);
   ObjectSetInteger(0, g_panelPrefix + "GradeVal", OBJPROP_COLOR, GradeColor(grade));

   int scoreWidth = (int)(g_currentSignal.setup_score * 140);
   if(scoreWidth < 1) scoreWidth = 1;
   ObjectSetInteger(0, g_panelPrefix + "ScoreBar", OBJPROP_XSIZE, scoreWidth);
   ObjectSetString (0, g_panelPrefix + "ScoreVal", OBJPROP_TEXT,
                    DoubleToString(g_currentSignal.setup_score, 2));

   // TP / SL probability bars
   int tpWidth = (int)(g_currentSignal.tp_probability * 140);
   int slWidth = (int)(g_currentSignal.sl_probability * 140);
   if(tpWidth < 1) tpWidth = 1;
   if(slWidth < 1) slWidth = 1;
   ObjectSetInteger(0, g_panelPrefix + "TPBar", OBJPROP_XSIZE, tpWidth);
   ObjectSetInteger(0, g_panelPrefix + "SLBar", OBJPROP_XSIZE, slWidth);
   ObjectSetString (0, g_panelPrefix + "TPVal", OBJPROP_TEXT, DoubleToString(g_currentSignal.tp_probability * 100, 0) + "%");
   ObjectSetString (0, g_panelPrefix + "SLVal", OBJPROP_TEXT, DoubleToString(g_currentSignal.sl_probability * 100, 0) + "%");

   ObjectSetString(0, g_panelPrefix + "ConfidenceVal", OBJPROP_TEXT,
                   DoubleToString(g_currentSignal.confidence * 100, 1) + "%");

   double rr = g_currentSignal.rr_planned > 0 ? g_currentSignal.rr_planned : InpRiskRewardRatio;
   ObjectSetString(0, g_panelPrefix + "RRVal", OBJPROP_TEXT,
                   "1:" + DoubleToString(rr, 1));

   ObjectSetString(0, g_panelPrefix + "SLKindVal", OBJPROP_TEXT,
                   g_currentSignal.sl_kind == "" ? "---" : g_currentSignal.sl_kind);
   ObjectSetString(0, g_panelPrefix + "TPKindVal", OBJPROP_TEXT,
                   g_currentSignal.tp_kind == "" ? "---" : g_currentSignal.tp_kind);

   ObjectSetString (0, g_panelPrefix + "VolRegVal", OBJPROP_TEXT,
                    g_currentSignal.volatility_regime == "" ? "---" : g_currentSignal.volatility_regime);
   ObjectSetInteger(0, g_panelPrefix + "VolRegVal", OBJPROP_COLOR,
                    VolRegimeColor(g_currentSignal.volatility_regime));

   ObjectSetString(0, g_panelPrefix + "SLDistVal", OBJPROP_TEXT,
                   DoubleToString(g_currentSignal.sl_distance, 1) + " pips");

   ObjectSetString(0, g_panelPrefix + "MethodVal", OBJPROP_TEXT,
                   g_currentSignal.method == "" ? "---" : g_currentSignal.method);

   ObjectSetString(0, g_panelPrefix + "HoldVal", OBJPROP_TEXT,
                   g_currentSignal.hold_reason == "" ? "---" : g_currentSignal.hold_reason);

   // -------- TRADE INFO --------
   if(g_tradeInfo.hasPosition)
   {
      ObjectSetString (0, g_panelPrefix + "StatusVal", OBJPROP_TEXT, "IN TRADE");
      ObjectSetInteger(0, g_panelPrefix + "StatusVal", OBJPROP_COLOR, CLR_POSITIVE);

      int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
      ObjectSetString(0, g_panelPrefix + "EntryVal",    OBJPROP_TEXT, DoubleToString(g_tradeInfo.entryPrice, digits));
      ObjectSetString(0, g_panelPrefix + "TradeSLVal",  OBJPROP_TEXT, DoubleToString(g_tradeInfo.slPrice, digits));
      ObjectSetString(0, g_panelPrefix + "TradeTPVal",  OBJPROP_TEXT, DoubleToString(g_tradeInfo.tpPrice, digits));
      ObjectSetString(0, g_panelPrefix + "RMultVal",    OBJPROP_TEXT, DoubleToString(g_tradeInfo.currentRMultiple, 2) + "R");
      ObjectSetString(0, g_panelPrefix + "PLVal",       OBJPROP_TEXT, DoubleToString(g_tradeInfo.floatingPL, 2));
      ObjectSetInteger(0, g_panelPrefix + "PLVal",      OBJPROP_COLOR,
                       g_tradeInfo.floatingPL >= 0 ? CLR_POSITIVE : CLR_NEGATIVE);
      ObjectSetString(0, g_panelPrefix + "EntryConfVal",OBJPROP_TEXT,
                      DoubleToString(g_tradeInfo.entryConfidence * 100, 1) + "%");
      int durMin = (int)((TimeCurrent() - g_tradeInfo.entryTime) / 60);
      ObjectSetString(0, g_panelPrefix + "TimeVal", OBJPROP_TEXT, IntegerToString(durMin) + "m");
   }
   else
   {
      ObjectSetString (0, g_panelPrefix + "StatusVal", OBJPROP_TEXT, "NO POSITION");
      ObjectSetInteger(0, g_panelPrefix + "StatusVal", OBJPROP_COLOR, CLR_TEXT_DIM);
      ObjectSetString(0, g_panelPrefix + "EntryVal",    OBJPROP_TEXT, "---");
      ObjectSetString(0, g_panelPrefix + "TradeSLVal",  OBJPROP_TEXT, "---");
      ObjectSetString(0, g_panelPrefix + "TradeTPVal",  OBJPROP_TEXT, "---");
      ObjectSetString(0, g_panelPrefix + "RMultVal",    OBJPROP_TEXT, "---");
      ObjectSetString(0, g_panelPrefix + "PLVal",       OBJPROP_TEXT, "---");
      ObjectSetInteger(0, g_panelPrefix + "PLVal",      OBJPROP_COLOR, CLR_TEXT_DIM);
      ObjectSetString(0, g_panelPrefix + "EntryConfVal",OBJPROP_TEXT, "---");
      ObjectSetString(0, g_panelPrefix + "TimeVal",     OBJPROP_TEXT, "---");
   }
}

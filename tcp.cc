/*
 * Copyright (c) 2025 MOSAFER
 * TCP CONGESTION CONTROL WITH OPENGYM INTEGRATION
 */

#include <fstream>
#include <string>
#include <algorithm>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h"
#include "ns3/opengym-module.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-westwood-plus.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpOpenGymExample");

class MyApp;

// ============================================================================
// MyApp Class
// ============================================================================
class MyApp : public Application
{
public:
  MyApp () : m_socket (0), m_running (false), m_packetsSent (0) {}
  virtual ~MyApp() { m_socket = 0; }

  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
  }

  void ChangeRate(DataRate newrate) { m_dataRate = newrate; }

private:
  virtual void StartApplication (void) {
    m_running = true;
    m_socket->Bind ();
    m_socket->Connect (m_peer);
    SendPacket ();
  }

  virtual void StopApplication (void) {
    m_running = false;
    if (m_sendEvent.IsRunning ()) Simulator::Cancel (m_sendEvent);
    if (m_socket) m_socket->Close ();
  }

  void SendPacket (void) {
    Ptr<Packet> packet = Create<Packet> (m_packetSize);
    m_socket->Send (packet);
    if (++m_packetsSent < m_nPackets) ScheduleTx ();
  }

  void ScheduleTx (void) {
    if (m_running) {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
    }
  }

  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
};

// ============================================================================
// OpenGym Environment
// ============================================================================
namespace ns3 {

class TcpOpenGymEnv : public OpenGymEnv
{
public:
  TcpOpenGymEnv () {
    m_simulationTime = 50.0;
    m_envStepTime = 0.1;
    m_currentCwnd = 0;
    m_previousCwnd = 0;
    m_throughput = 0.0;
    m_packetLoss = 0;
    m_lastTotalRx = 0;
    m_currentRtt = Seconds(0);
  }

  virtual ~TcpOpenGymEnv () {}

  static TypeId GetTypeId (void) {
    static TypeId tid = TypeId ("ns3::TcpOpenGymEnv")
      .SetParent<OpenGymEnv>()
      .SetGroupName("OpenGym")
      .AddConstructor<TcpOpenGymEnv>();
    return tid;
  }

  void SetupEnv(double simTime, double stepTime) {
    m_simulationTime = simTime;
    m_envStepTime = stepTime;
  }

  void SetApp(Ptr<MyApp> app, DataRate initialRate) {
    m_app = app;
    m_currentRate = initialRate;
  }

  void SetSink(Ptr<PacketSink> sink) { m_sink = sink; }

  Ptr<OpenGymSpace> GetActionSpace() override { return CreateObject<OpenGymDiscreteSpace> (5); }

  Ptr<OpenGymSpace> GetObservationSpace() override {
    std::vector<uint32_t> shape = {5,};
    return CreateObject<OpenGymBoxSpace> (0.0, 1.0, shape, TypeNameGet<float>());
  }

  bool GetGameOver() override {
    bool isGameOver = (Simulator::Now ().GetSeconds () >= m_simulationTime);
    NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
    return isGameOver;
  }

  Ptr<OpenGymDataContainer> GetObservation() override {
    std::vector<uint32_t> shape = {5,};
    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);

    float normCwnd = std::min(1.0f, (float)m_currentCwnd / 10000000.0f);
    float normRtt = std::min(1.0f, (float)m_currentRtt.GetSeconds() / 0.1f);
    float normThroughput = std::min(1.0f, (float)m_throughput / 5000000.0f);
    float normLoss = std::min(1.0f, (float)m_packetLoss / 100.0f);

    box->AddValue(normCwnd);
    box->AddValue(normRtt);
    box->AddValue(normThroughput);
    box->AddValue(normLoss); 
    box->AddValue(0.5f); 

    NS_LOG_UNCOND ("MyGetObservation: [" << normCwnd << ", " << normRtt << ", " 
                   << normThroughput << ", " << normLoss << ", 0.5]");
    return box;
  }
/*
  float GetReward() override {
    float thrMbps = m_throughput / 1000000.0f;
    float reward = thrMbps * 3.0f; 
    reward -= (m_currentRtt.GetSeconds() * 10.0f); // Latency penalty
    reward -= (m_packetLoss * 2.0f);              // Loss penalty


    NS_LOG_UNCOND ("MyGetReward: " << reward << " (throughput=" << thrMbps 
                   << " Mbps, rtt=" << m_currentRtt.GetSeconds() << " s, loss=" << m_packetLoss << ")");
    return reward;
  }
  */
float GetReward() override {
    // 1. MATCH YOUR ACTUAL BOTTLENECK (1.5 Mbps)
    const float MAX_CAPACITY_MBPS = 1.5f; 
    const float TARGET_RTT_SECONDS = 0.025f; // 25ms is a realistic target for a 2ms link
    
    float thrMbps = m_throughput / 1000000.0f;
    
    // Normalize Throughput (0.0 to 1.0)
    float normalizedThr = std::min(thrMbps / MAX_CAPACITY_MBPS, 1.0f);
    
    // Normalize RTT (Penalize only if it exceeds target)
    float rttSeconds = m_currentRtt.GetSeconds();
    float normalizedRtt = 0.0f;
    if (rttSeconds > TARGET_RTT_SECONDS) {
        normalizedRtt = (rttSeconds - TARGET_RTT_SECONDS) / 0.1f; // Scale by 100ms
    }
    
    // Use LOSS COUNT for THIS STEP ONLY
    float lossPenalty = (float)m_packetLoss * 1.0f; 
    
    // RESET loss for the next step so we don't punish the agent forever
    m_packetLoss = 0; 

    // Balanced Weights
    // Throughput is the goal (+1.0), RTT/Loss are the constraints (-0.5)
    float reward = (normalizedThr * 1.0f) - (normalizedRtt * 0.5f) - (lossPenalty * 0.5f);
    
    // Keep it in a range DQN likes
    reward = std::max(std::min(reward, 2.0f), -5.0f);
    
    return reward;
}
/*
float GetReward() override {
    // 1. Calculate Throughput in Mbps
    // Use std::max to ensure throughput is at least a tiny positive value (e.g., 1 bps)
    double thrMbps = std::max(1.0, m_throughput) / 1000000.0;

    // 2. Get RTT in seconds
    // Use std::max to ensure RTT is at least 0.1ms to avoid log(0)
    double rttSec = std::max(0.0001, m_currentRtt.GetSeconds());

    // 3. Compute log-based reward
    // This rewards higher throughput and penalizes higher RTT exponentially
    float reward = std::log(thrMbps) - std::log(rttSec);

    // Optional: Log the values for debugging
    NS_LOG_UNCOND ("MyGetReward: " << reward 
                   << " (thr=" << thrMbps << " Mbps"
                   << ", rtt=" << rttSec << " s)");

    return reward;
}
*/


  // FIXED: Re-formatted for Python TrainingMonitor parser
  std::string GetExtraInfo() override {
    std::string info = "cwnd=" + std::to_string(m_currentCwnd) + 
                       ", rtt=" + std::to_string(m_currentRtt.GetSeconds()) +
                       ", throughput=" + std::to_string(m_throughput/1000000.0f) + " Mbps" +
                       ", loss=" + std::to_string(m_packetLoss);
    NS_LOG_UNCOND("MyGetExtraInfo: " << info);
    return info;
  }

  bool ExecuteActions(Ptr<OpenGymDataContainer> action) override {
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    uint32_t actionValue = discrete->GetValue();
    NS_LOG_UNCOND ("ExecuteActions: " << actionValue);

    double multiplier = 1.0;
    if (actionValue == 0) multiplier = 0.5;
    else if (actionValue == 1) multiplier = 0.75;
    else if (actionValue == 3) multiplier = 1.25;
    else if (actionValue == 4) multiplier = 1.5;

    m_currentRate = DataRate(m_currentRate.GetBitRate() * multiplier);
    if (m_currentRate.GetBitRate() < 100000) m_currentRate = DataRate("100Kbps");
    if (m_currentRate.GetBitRate() > 10000000) m_currentRate = DataRate("10Mbps");

    if (m_app) m_app->ChangeRate(m_currentRate);
    
    ScheduleNextStateRead();
    return true;
  }

  void ScheduleNextStateRead() {
    Simulator::Schedule (Seconds(m_envStepTime), &TcpOpenGymEnv::GetCurrentState, this);
  }

  void GetCurrentState() {
    if (m_sink) {
      uint64_t totalRx = m_sink->GetTotalRx();
      Time now = Simulator::Now();
      if (m_lastUpdateTime > Seconds(0)) {
        double timeDiff = (now - m_lastUpdateTime).GetSeconds();
        if (timeDiff > 0) m_throughput = (totalRx - m_lastTotalRx) * 8.0 / timeDiff;
      }
      m_lastTotalRx = totalRx;
      m_lastUpdateTime = now;
    }
    Notify();
  }

  void SetCwnd(uint32_t oldCwnd, uint32_t newCwnd) { m_currentCwnd = newCwnd; }
  void SetRtt(Time oldRtt, Time newRtt) { m_currentRtt = newRtt; }
  void IncrementLoss() { m_packetLoss++; }

private:
  Ptr<MyApp> m_app;
  DataRate m_currentRate;
  uint32_t m_currentCwnd;
  uint32_t m_previousCwnd;
  Time m_currentRtt;
  double m_throughput;
  uint32_t m_packetLoss;
  double m_simulationTime;
  double m_envStepTime;
  Ptr<PacketSink> m_sink;
  uint64_t m_lastTotalRx;
  Time m_lastUpdateTime;
};

// Tracers
static void CwndTracer (Ptr<TcpOpenGymEnv> env, uint32_t oldCwnd, uint32_t newCwnd) { env->SetCwnd(oldCwnd, newCwnd); }
static void RttTracer (Ptr<TcpOpenGymEnv> env, Time oldRtt, Time newRtt) { env->SetRtt(oldRtt, newRtt); }
static void DeviceDropTracer (Ptr<TcpOpenGymEnv> env, Ptr<const Packet> p) { env->IncrementLoss(); }

} // namespace ns3

// ============================================================================
// Main
// ============================================================================
int main (int argc, char* argv[])
{
  std::string lat = "10ms";
  std::string rate = "10Mbps";     
  std::string rate1 = "1.5Mbps";   // Bottleneck
  double simTime = 50.0;
  double envStepTime = 0.1;
  uint32_t openGymPort = 5555;
  bool openGymEnabled = true;

  CommandLine cmd;
  cmd.AddValue ("openGym", "Enable OpenGym", openGymEnabled);
  cmd.AddValue ("simTime", "Total simulation time", simTime);
  cmd.AddValue ("envStepTime", "Step time", envStepTime);
  cmd.AddValue ("openGymPort", "Port", openGymPort);
  cmd.Parse (argc, argv);

  NodeContainer c;
  c.Create(8);
  InternetStackHelper internet;
  internet.Install (c);

  PointToPointHelper p2p, p2p_bottleneck;
  p2p.SetDeviceAttribute ("DataRate", StringValue (rate));
  p2p.SetChannelAttribute ("Delay", StringValue (lat));
  p2p_bottleneck.SetDeviceAttribute ("DataRate", StringValue (rate1));
  p2p_bottleneck.SetChannelAttribute ("Delay", StringValue (lat));
  p2p_bottleneck.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("20p"));

  NetDeviceContainer d0d3 = p2p.Install (c.Get(0), c.Get(3));
  NetDeviceContainer d1d3 = p2p.Install (c.Get(1), c.Get(3));
  NetDeviceContainer d2d3 = p2p.Install (c.Get(2), c.Get(3));
  NetDeviceContainer d3d4 = p2p_bottleneck.Install (c.Get(3), c.Get(4)); 
  NetDeviceContainer d5d4 = p2p.Install (c.Get(5), c.Get(4));
  NetDeviceContainer d6d4 = p2p.Install (c.Get(6), c.Get(4));
  NetDeviceContainer d7d4 = p2p.Install (c.Get(7), c.Get(4));

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0"); ipv4.Assign (d0d3);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0"); ipv4.Assign (d1d3);
  ipv4.SetBase ("10.1.3.0", "255.255.255.0"); ipv4.Assign (d2d3);
  ipv4.SetBase ("10.1.4.0", "255.255.255.0"); Ipv4InterfaceContainer i3i4 = ipv4.Assign (d3d4);
  ipv4.SetBase ("10.1.5.0", "255.255.255.0"); Ipv4InterfaceContainer i5i4 = ipv4.Assign (d5d4);
  ipv4.SetBase ("10.1.6.0", "255.255.255.0"); Ipv4InterfaceContainer i6i4 = ipv4.Assign (d6d4);
  ipv4.SetBase ("10.1.7.0", "255.255.255.0"); Ipv4InterfaceContainer i7i4 = ipv4.Assign (d7d4);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Ptr<TcpOpenGymEnv> openGymEnv = CreateObject<TcpOpenGymEnv>();
  openGymEnv->SetupEnv(simTime, envStepTime);
  openGymEnv->SetOpenGymInterface(OpenGymInterface::Get(openGymPort));

  // Connect drop tracer to the bottleneck queue
  d3d4.Get(0)->TraceConnectWithoutContext("PhyRxDrop", MakeBoundCallback(&ns3::DeviceDropTracer, openGymEnv));

  uint16_t port0 = 8080;
  PacketSinkHelper sinkH0 ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port0));
  ApplicationContainer sinkApp0 = sinkH0.Install (c.Get (5));
  sinkApp0.Start (Seconds (1.));
  openGymEnv->SetSink(DynamicCast<PacketSink>(sinkApp0.Get(0)));

  Ptr<Socket> socket0 = Socket::CreateSocket (c.Get (0), TcpSocketFactory::GetTypeId ());
  socket0->TraceConnectWithoutContext("CongestionWindow", MakeBoundCallback(&ns3::CwndTracer, openGymEnv));
  socket0->TraceConnectWithoutContext("RTT", MakeBoundCallback(&ns3::RttTracer, openGymEnv));

  Ptr<MyApp> app0 = CreateObject<MyApp> ();
  app0->Setup (socket0, InetSocketAddress (i5i4.GetAddress (0), port0), 1040, 1000000, DataRate ("1Mbps"));
  c.Get (0)->AddApplication (app0);
  app0->SetStartTime (Seconds (1.));
  openGymEnv->SetApp(app0, DataRate("1Mbps"));

  // BG Traffic
  uint16_t port1 = 8081;
  PacketSinkHelper sinkH1 ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port1));
  sinkH1.Install (c.Get (6)).Start(Seconds(1.));
  Ptr<Socket> socket1 = Socket::CreateSocket (c.Get (1), TcpSocketFactory::GetTypeId ());
  Ptr<MyApp> app1 = CreateObject<MyApp> ();
  app1->Setup (socket1, InetSocketAddress (i6i4.GetAddress (0), port1), 1040, 1000000, DataRate ("1Mbps"));
  c.Get (1)->AddApplication (app1);
  app1->SetStartTime (Seconds (2.));
  app1->SetStopTime (Seconds (10.));

  uint16_t port2 = 8082;
  PacketSinkHelper sinkH2 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port2));
  sinkH2.Install (c.Get (7)).Start(Seconds(1.));
  Ptr<Socket> socket2 = Socket::CreateSocket (c.Get (2), UdpSocketFactory::GetTypeId ());
  Ptr<MyApp> app2 = CreateObject<MyApp> ();
  app2->Setup (socket2, InetSocketAddress (i7i4.GetAddress (0), port2), 1040, 1000000, DataRate ("1Mbps"));
  c.Get (2)->AddApplication (app2);
  app2->SetStartTime (Seconds (3.));
  app2->SetStopTime (Seconds (10.));

  if (openGymEnabled) {
    Simulator::Schedule (Seconds(envStepTime), &TcpOpenGymEnv::ScheduleNextStateRead, openGymEnv);
  }

  Simulator::Stop (Seconds(simTime));
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
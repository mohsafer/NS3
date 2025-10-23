
/*

TCP CONGESTION ALGORITHMS
By
 __  __   ___   ____      _     _____  _____  ____
|  \/  | / _ \ / ___|    / \   |  ___|| ____||  _ \
| |\/| || | | |\___ \   / _ \  | |_   |  _|  | |_) |
| |  | || |_| | ___) | / ___ \ |  _|  | |___ |  _ <
|_|  |_| \___/ |____/ /_/   \_\|_|    |_____||_| \_\


             Network Topology

 N0----                            ----N5
       |          (p2p)           |
 N1---------N3 <--------> N4-----------N6
       |                          |
 N2----                            ----N7

*/

#include <fstream>
#include <string>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h"
#include "ns3/opengym-module.h"

using namespace ns3;


NS_LOG_COMPONENT_DEFINE ("Assignment");

class MyApp : public Application
{
public:

  MyApp ();
  virtual ~MyApp();

  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);
  void ChangeRate(DataRate newrate);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);

  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
};

MyApp::MyApp ()
  : m_socket (0),
    m_peer (),
    m_packetSize (0),
    m_nPackets (0),
    m_dataRate (0),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0)
{
}

MyApp::~MyApp()
{
  m_socket = 0;
}

void
MyApp::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
}

void
MyApp::StartApplication (void)
{
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind ();
  m_socket->Connect (m_peer);
  SendPacket ();
}

void
MyApp::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}

void
MyApp::SendPacket (void)
{
  Ptr<Packet> packet = Create<Packet> (m_packetSize);
  m_socket->Send (packet);

  if (++m_packetsSent < m_nPackets)
    {
      ScheduleTx ();
    }
}

void
MyApp::ScheduleTx (void)
{
  if (m_running)
    {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
    }
}

void
MyApp::ChangeRate(DataRate newrate)
{
   m_dataRate = newrate;
   return;
}

static void
CwndChange (uint32_t oldCwnd, uint32_t newCwnd)
{
  std::cout << Simulator::Now ().GetSeconds () << "\t" << newCwnd <<"\n";
}

void
IncRate (Ptr<MyApp> app, DataRate rate)
{
	app->ChangeRate(rate);
    return;
}

// ============================================================================
// OpenGym Environment for DQN-based TCP Congestion Control
// ============================================================================

namespace ns3 {

class TcpOpenGymEnv : public OpenGymEnv
{
public:
  TcpOpenGymEnv ();
  TcpOpenGymEnv (uint32_t simSeed, double simulationTime, uint32_t openGymPort, double envStepTime);
  virtual ~TcpOpenGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  // OpenGym interface
  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymSpace> GetObservationSpace();
  bool GetGameOver();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);

  // Environment-specific methods
  void ScheduleNextStateRead();
  void SetCwnd(uint32_t oldCwnd, uint32_t newCwnd);
  void SetRtt(Time oldRtt, Time newRtt);
  void SetThroughput(double throughput);
  void SetPacketLoss(uint32_t lostPackets);

private:
  void GetCurrentState();

  // Simulation parameters
  uint32_t m_simSeed;
  double m_simulationTime;
  double m_envStepTime;
  uint32_t m_openGymPort;

  // Network state variables
  uint32_t m_currentCwnd;
  uint32_t m_previousCwnd;
  Time m_currentRtt;
  Time m_previousRtt;
  double m_throughput;
  uint32_t m_packetLoss;
  uint32_t m_totalPacketsSent;
  uint32_t m_totalPacketsReceived;

  // RL state tracking
  bool m_gameOver;
  double m_reward;
  std::vector<double> m_observation;
};

TcpOpenGymEnv::TcpOpenGymEnv ()
{
  NS_LOG_FUNCTION (this);
  m_simSeed = 1;
  m_simulationTime = 50.0;
  m_envStepTime = 0.1; // Take RL action every 0.1 seconds
  m_openGymPort = 5555;
  m_gameOver = false;
  m_currentCwnd = 0;
  m_previousCwnd = 0;
  m_throughput = 0.0;
  m_packetLoss = 0;
  m_totalPacketsSent = 0;
  m_totalPacketsReceived = 0;
  m_reward = 0.0;
}

TcpOpenGymEnv::TcpOpenGymEnv (uint32_t simSeed, double simulationTime, uint32_t openGymPort, double envStepTime)
{
  NS_LOG_FUNCTION (this);
  m_simSeed = simSeed;
  m_simulationTime = simulationTime;
  m_openGymPort = openGymPort;
  m_envStepTime = envStepTime;
  m_gameOver = false;
  m_currentCwnd = 0;
  m_previousCwnd = 0;
  m_throughput = 0.0;
  m_packetLoss = 0;
  m_totalPacketsSent = 0;
  m_totalPacketsReceived = 0;
  m_reward = 0.0;
}

TcpOpenGymEnv::~TcpOpenGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
TcpOpenGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpOpenGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<TcpOpenGymEnv> ()
  ;
  return tid;
}

void
TcpOpenGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

/*
Define action space: Discrete actions for congestion control
Actions:
  0 - Decrease sending rate by 50%
  1 - Decrease sending rate by 25%
  2 - Maintain current rate
  3 - Increase sending rate by 25%
  4 - Increase sending rate by 50%
*/
Ptr<OpenGymSpace>
TcpOpenGymEnv::GetActionSpace()
{
  uint32_t actionNum = 5;
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (actionNum);
  NS_LOG_UNCOND ("GetActionSpace: " << space);
  return space;
}

/*
Define observation space: Continuous observations
Observations (normalized values):
  0 - Current Congestion Window (normalized)
  1 - Current RTT (normalized)
  2 - Throughput (normalized)
  3 - Packet Loss Rate (normalized)
  4 - CWND change rate
*/
Ptr<OpenGymSpace>
TcpOpenGymEnv::GetObservationSpace()
{
  float low = 0.0;
  float high = 1.0;
  std::vector<uint32_t> shape = {5,};
  std::string dtype = TypeNameGet<float> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_UNCOND ("GetObservationSpace: " << space);
  return space;
}

bool
TcpOpenGymEnv::GetGameOver()
{
  bool isGameOver = (Simulator::Now ().GetSeconds () >= m_simulationTime);
  m_gameOver = isGameOver;
  NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
  return m_gameOver;
}

Ptr<OpenGymDataContainer>
TcpOpenGymEnv::GetObservation()
{
  std::vector<uint32_t> shape = {5,};
  Ptr<OpenGymBoxContainer<float> > box = CreateObject<OpenGymBoxContainer<float> >(shape);

  // Normalize observations to [0, 1] range
  // CWND normalized (assuming max CWND ~ 1000)
  float normCwnd = std::min(1.0f, (float)m_currentCwnd / 1000.0f);
  
  // RTT normalized (assuming max RTT ~ 1 second)
  float normRtt = std::min(1.0f, (float)m_currentRtt.GetSeconds());
  
  // Throughput normalized (assuming max throughput ~ 10 Mbps)
  float normThroughput = std::min(1.0f, (float)m_throughput / 10000000.0f);
  
  // Packet loss rate
  float lossRate = 0.0f;
  if (m_totalPacketsSent > 0) {
    lossRate = std::min(1.0f, (float)m_packetLoss / (float)m_totalPacketsSent);
  }
  
  // CWND change rate
  float cwndChangeRate = 0.0f;
  if (m_previousCwnd > 0) {
    cwndChangeRate = std::min(1.0f, std::max(-1.0f, 
      (float)(m_currentCwnd - m_previousCwnd) / (float)m_previousCwnd));
    cwndChangeRate = (cwndChangeRate + 1.0f) / 2.0f; // Normalize to [0, 1]
  }

  box->AddValue(normCwnd);
  box->AddValue(normRtt);
  box->AddValue(normThroughput);
  box->AddValue(lossRate);
  box->AddValue(cwndChangeRate);

  NS_LOG_UNCOND ("MyGetObservation: " << box);
  return box;
}

float
TcpOpenGymEnv::GetReward()
{
  // Reward function balancing throughput and fairness
  // Reward = throughput - penalty_for_loss - penalty_for_delay
  
  float reward = 0.0f;
  
  // Positive reward for throughput (Mbps)
  reward += m_throughput / 1000000.0f;
  
  // Penalty for packet loss
  float lossRate = 0.0f;
  if (m_totalPacketsSent > 0) {
    lossRate = (float)m_packetLoss / (float)m_totalPacketsSent;
  }
  reward -= lossRate * 10.0f; // Heavy penalty for packet loss
  
  // Penalty for high RTT
  reward -= m_currentRtt.GetSeconds() * 2.0f;
  
  NS_LOG_UNCOND ("MyGetReward: " << reward);
  return reward;
}

std::string
TcpOpenGymEnv::GetExtraInfo()
{
  std::string myInfo = "cwnd=" + std::to_string(m_currentCwnd) + 
                       ", rtt=" + std::to_string(m_currentRtt.GetSeconds()) +
                       ", throughput=" + std::to_string(m_throughput) +
                       ", loss=" + std::to_string(m_packetLoss);
  NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

bool
TcpOpenGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t actionValue = discrete->GetValue();
  
  NS_LOG_UNCOND ("ExecuteActions: " << actionValue);
  
  // Actions will be applied to modify the sending rate
  // This would typically modify application sending rate or TCP parameters
  // The actual implementation depends on how you want to control the network
  
  // Schedule next state observation
  ScheduleNextStateRead();
  
  return true;
}

void
TcpOpenGymEnv::ScheduleNextStateRead()
{
  Simulator::Schedule (Seconds(m_envStepTime), &TcpOpenGymEnv::GetCurrentState, this);
}

void
TcpOpenGymEnv::GetCurrentState()
{
  NS_LOG_FUNCTION (this);
  
  // Notify OpenGym that new state is ready
  Notify();
}

void
TcpOpenGymEnv::SetCwnd(uint32_t oldCwnd, uint32_t newCwnd)
{
  m_previousCwnd = m_currentCwnd;
  m_currentCwnd = newCwnd;
}

void
TcpOpenGymEnv::SetRtt(Time oldRtt, Time newRtt)
{
  m_previousRtt = m_currentRtt;
  m_currentRtt = newRtt;
}

void
TcpOpenGymEnv::SetThroughput(double throughput)
{
  m_throughput = throughput;
}

void
TcpOpenGymEnv::SetPacketLoss(uint32_t lostPackets)
{
  m_packetLoss = lostPackets;
}

} // namespace ns3

// ============================================================================
// End of OpenGym Environment
// ============================================================================


int main (int argc, char* argv[])
{

  std::string lat = "2ms";
  std::string rate = "5Mbps";		 // P2P link
  std::string rate1="2Mbps"; 		// for Node 3 to 4
  std::string tcpVariant="TcpVegas";
  bool enableFlowMonitor = false;
  
  // OpenGym parameters
  bool openGymEnabled = true;
  uint32_t openGymPort = 5555;
  double envStepTime = 0.1; // RL agent decision interval
  uint32_t simSeed = 1;
  double simTime = 50.0;

  //*********************************************************************************************
  //********************** Specifying  TCP Congestion control Algorithm. ************************
  //  Call any TCP type via Command line instead of changing the source
  // ./ns3 run "program" -tcpVariant=Tcpvegas"
  //********************************************************************************************


  CommandLine cmd;
  cmd.AddValue ("latency", "P2P link Latency in seconds", lat);
  cmd.AddValue ("rate", "P2P data rate in bps", rate);
  cmd.AddValue ("EnableMonitor", "Enable Flow Monitor", enableFlowMonitor);
  cmd.AddValue("tcpVariant",
                 "Transport protocol to use: TcpNewReno, "
                 "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                 "TcpBic,TcpCubic, TcpYeah, TcpIllinois, TcpWestwood, TcpWestwoodPlus, TcpLedbat ",
                 tcpVariant);
  cmd.AddValue ("openGym", "Enable OpenGym for RL", openGymEnabled);
  cmd.AddValue ("openGymPort", "Port number for OpenGym", openGymPort);
  cmd.AddValue ("envStepTime", "Time step for environment updates", envStepTime);
  cmd.AddValue ("simSeed", "Seed for random variables", simSeed);
  cmd.AddValue ("simTime", "Total simulation time", simTime);
  cmd.Parse (argc, argv);



  //Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
  //Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpVegas::GetTypeId()));


    cmd.Parse(argc, argv);

    // Set random seed for reproducibility
    RngSeedManager::SetSeed(simSeed);

    tcpVariant = std::string("ns3::") + tcpVariant;
    // Select TCP variant
    if (tcpVariant == "ns3::TcpWestwoodPlus")
    {
        // TcpWestwoodPlus is not an actual TypeId name; we need TcpWestwood here
        Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpWestwood::GetTypeId()));
        // the default protocol type in ns3::TcpWestwood is WESTWOOD
        Config::SetDefault("ns3::TcpWestwood::ProtocolType", EnumValue(TcpWestwood::WESTWOODPLUS));
    }
    else
    {
        TypeId tcpTid;
        NS_ABORT_MSG_UNLESS(TypeId::LookupByNameFailSafe(tcpVariant, &tcpTid),
                            "TypeId " << tcpVariant << " not found");
        Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                           TypeIdValue(TypeId::LookupByName(tcpVariant)));
    }



/// ***************************Implementation of Classic Main function
/*
int main ()
{

  std::string protocol = "TcpNewReno";
  std::string lat = "2ms";
  std::string rate = "5Mbps";		 // P2P link
  std::string rate1="2Mbps"; 		// for Node 3 to 4
  bool enableFlowMonitor = false;


  if (protocol == "TcpNewReno")
   Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpNewReno"));
  else
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpVeno"));

//Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue (TcpTahoe::GetTypeId()));


  CommandLine cmd;
  cmd.AddValue ("latency", "P2P link Latency in miliseconds", lat);
  cmd.AddValue ("rate", "P2P data rate in bps", rate);
  cmd.AddValue ("EnableMonitor", "Enable Flow Monitor", enableFlowMonitor);

  //cmd.Parse (argc, argv);


//Sets the default congestion control algorithm
  //Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpTahoe"));

*/

//***************** Nodes Creation required by the topology as Shown above********************

  NS_LOG_INFO ("Create nodes.");
  NodeContainer c; // ALL Nodes
  c.Create(8);

  NodeContainer n0n3 = NodeContainer (c.Get (0), c.Get (3));
  NodeContainer n1n3 = NodeContainer (c.Get (1), c.Get (3));
  NodeContainer n2n3 = NodeContainer (c.Get (2), c.Get (3));
  NodeContainer n3n4 = NodeContainer (c.Get (3), c.Get (4));
  NodeContainer n5n4 = NodeContainer (c.Get (5), c.Get (4));
  NodeContainer n6n4 = NodeContainer (c.Get (6), c.Get (4));
  NodeContainer n7n4 = NodeContainer (c.Get (7), c.Get (4));

  // OpenGym Environment Setup
  Ptr<TcpOpenGymEnv> openGymEnv = nullptr;
  if (openGymEnabled)
    {
      NS_LOG_UNCOND("Initializing OpenGym environment...");
      openGymEnv = CreateObject<TcpOpenGymEnv> (simSeed, simTime, openGymPort, envStepTime);
      openGymEnv->SetOpenGymInterface(OpenGymInterface::Get(openGymPort));
      NS_LOG_UNCOND("OpenGym interface ready. Connect Python RL agent to port " << openGymPort);
    }



//************************ Install Internet Stack*********************************

  InternetStackHelper internet;
  internet.Install (c);

//**************** channels Creation without IP addressing*************************

  NS_LOG_INFO ("Create channels.");
  PointToPointHelper p2p,p2p_for3_4;
  p2p.SetDeviceAttribute ("DataRate", StringValue (rate));
  p2p.SetChannelAttribute ("Delay", StringValue (lat));
  NetDeviceContainer d0d3 = p2p.Install (n0n3);
  NetDeviceContainer d1d3 = p2p.Install (n1n3);
  NetDeviceContainer d2d3 = p2p.Install (n2n3);
  NetDeviceContainer d5d4 = p2p.Install (n5n4);
  NetDeviceContainer d6d4 = p2p.Install (n6n4);
  NetDeviceContainer d7d4 = p2p.Install (n7n4);

  p2p_for3_4.SetDeviceAttribute ("DataRate", StringValue (rate1)); // for Node 3 to 5 data rate is 2Mbps
  p2p_for3_4.SetChannelAttribute ("Delay", StringValue (lat));
  NetDeviceContainer d3d4 = p2p_for3_4.Install (n3n4);


//*********************IP addresses Setup******************************************

  NS_LOG_INFO ("Assign IP Addresses.");

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i0i3 = ipv4.Assign (d0d3);

  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer i1i3 = ipv4.Assign (d1d3);

  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer i2i3 = ipv4.Assign (d2d3);

  ipv4.SetBase ("10.1.4.0", "255.255.255.0");
  Ipv4InterfaceContainer i3i4 = ipv4.Assign (d3d4);

  ipv4.SetBase ("10.1.5.0", "255.255.255.0");
  Ipv4InterfaceContainer i5i4 = ipv4.Assign (d5d4);

  ipv4.SetBase ("10.1.6.0", "255.255.255.0");
  Ipv4InterfaceContainer i6i4 = ipv4.Assign (d6d4);

  ipv4.SetBase ("10.1.7.0", "255.255.255.0");
  Ipv4InterfaceContainer i7i4 = ipv4.Assign (d7d4);

   NS_LOG_INFO ("Enable static global routing.");


  //***************** Turn on global static routing for routing across the network******************

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  NS_LOG_INFO ("Create Applications.");



  //************** TCP connection from N0 to N5**********************************************

  uint16_t sinkPort = 8080;
  Address sinkAddress (InetSocketAddress (i5i4.GetAddress (0), sinkPort)); // interface of n5
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
  ApplicationContainer sinkApps = packetSinkHelper.Install (c.Get (5)); //n5 as sink
  sinkApps.Start (Seconds (2.));
  //sinkApps.Stop (Seconds (25.));


  //Ptr<Socket> ns3TcpSocket1 = Socket::CreateSocket (c.Get (0), tid); //source at n0
  Ptr<Socket> ns3TcpSocket = Socket::CreateSocket (c.Get (0), TcpSocketFactory::GetTypeId ()); //source at n0

  //********************* Congestion window ******************************

  // Modified callback to update OpenGym environment
  if (openGymEnabled && openGymEnv)
    {
      ns3TcpSocket->TraceConnectWithoutContext ("CongestionWindow", 
        MakeBoundCallback (&TcpOpenGymEnv::SetCwnd, openGymEnv));
    }
  else
    {
      ns3TcpSocket->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));
    }

  //*********************TCP application at N0*******************************

  Ptr<MyApp> app = CreateObject<MyApp> ();
  app->Setup (ns3TcpSocket, sinkAddress, 1040, 100000, DataRate ("1Mbps"));
  c.Get (0)->AddApplication (app);
  app->SetStartTime (Seconds (2.));
  //app->SetStopTime (Seconds (25.)); // you can change this value if you want limits the duration of running apps



  //*************************** TCP connection from N1 to N6***********************
  uint16_t sinkPort2 = 8081;
 // std::ostringstream tcpModel;
  Address sinkAddress2 (InetSocketAddress (i6i4.GetAddress (1), sinkPort2)); // interface of n6
  PacketSinkHelper packetSinkHelper2 ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort2));
  ApplicationContainer sinkApps2 = packetSinkHelper2.Install (c.Get (6)); //n6 as sink
  sinkApps2.Start (Seconds (5.));
   // sinkApps2.Stop (Seconds (6.));

  Ptr<Socket> ns3TcpSocket2 = Socket::CreateSocket (c.Get (1), TcpSocketFactory::GetTypeId ()); //source at n1

  //*********************** Congestion window for N1 to N6 *************

  if (openGymEnabled && openGymEnv)
    {
      ns3TcpSocket2->TraceConnectWithoutContext ("CongestionWindow", 
        MakeBoundCallback (&TcpOpenGymEnv::SetCwnd, openGymEnv));
    }
  else
    {
      ns3TcpSocket2->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));
    }

  // **************************Create TCP application at N1 ******************

  Ptr<MyApp> app2 = CreateObject<MyApp> ();
  app2->Setup (ns3TcpSocket2, sinkAddress2, 1040, 100000, DataRate ("1Mbps"));
  c.Get (1)->AddApplication (app2);
  app2->SetStartTime (Seconds (5.));
  //app2->SetStopTime (Seconds (6.));


  // *********************UDP connection from N2 to N7 ****************************

  uint16_t sinkPort3 = 6;
  Address sinkAddress3 (InetSocketAddress (i7i4.GetAddress (0), sinkPort3)); // interface of n7
  PacketSinkHelper packetSinkHelper3 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort3));
  ApplicationContainer sinkApps3 = packetSinkHelper3.Install (c.Get (7)); //n7 as sink
  sinkApps3.Start (Seconds (10.));
  sinkApps3.Stop (Seconds (17.));

  Ptr<Socket> ns3UdpSocket = Socket::CreateSocket (c.Get (2), UdpSocketFactory::GetTypeId ()); //source at n2

  // ***************************Create UDP application at N2 ***********************

  Ptr<MyApp> app3 = CreateObject<MyApp> ();
  app3->Setup (ns3UdpSocket, sinkAddress3, 1040, 100000, DataRate ("1Mbps"));
  c.Get (2)->AddApplication (app3);
  app3->SetStartTime (Seconds (10.));
  app3->SetStopTime (Seconds (17.));

  //ns3UdpSocket->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));



 // **********************************Increase UDP Rate ******************************

 // Simulator::Schedule (Seconds(20.0), &IncRate, app3, DataRate("2Mbps"));

  // Flow Monitor
  Ptr<FlowMonitor> flowmon;
  if (enableFlowMonitor)
    {
      FlowMonitorHelper flowmonHelper;
      flowmon = flowmonHelper.InstallAll ();
    }



//
// actual simulation.
//

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Stop (Seconds(simTime));

   //Enabling Pcap Tracing
   //p2p.EnablePcapAll("scratch/Assignment");

  // Start OpenGym state collection
  if (openGymEnabled && openGymEnv)
    {
      Simulator::Schedule (Seconds(envStepTime), &TcpOpenGymEnv::ScheduleNextStateRead, openGymEnv);
    }

  AnimationInterface anim("Assignment.xml");
  anim.SetConstantPosition(c.Get(0),0.0,0.0);
  anim.SetConstantPosition(c.Get(1),0.0,2.0);
  anim.SetConstantPosition(c.Get(2),0.0,4.0);
  anim.SetConstantPosition(c.Get(3),2.0,2.0);
  anim.SetConstantPosition(c.Get(4),4.0,2.0);
  anim.SetConstantPosition(c.Get(5),6.0,0.0);
  anim.SetConstantPosition(c.Get(6),6.0,2.0);
  anim.SetConstantPosition(c.Get(7),6.0,4.0);

  Simulator::Run ();
  if (enableFlowMonitor)
    {
	  flowmon->CheckForLostPackets ();
	  flowmon->SerializeToXmlFile("Assignment.flowmon", true, true);
    }
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}

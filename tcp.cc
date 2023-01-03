
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

int main (int argc, char* argv[])
{

  std::string lat = "2ms";
  std::string rate = "5Mbps";		 // P2P link
  std::string rate1="2Mbps"; 		// for Node 3 to 4
  std::string tcpVariant="TcpVegas";
  bool enableFlowMonitor = false;

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
  cmd.Parse (argc, argv);



  //Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
  //Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpVegas::GetTypeId()));


    cmd.Parse(argc, argv);

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

  ns3TcpSocket->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));

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

  ns3TcpSocket2->TraceConnectWithoutContext ("CongestionWindow", MakeCallback (&CwndChange));

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
  Simulator::Stop (Seconds(50.0));

   //Enabling Pcap Tracing
   //p2p.EnablePcapAll("scratch/Assignment");


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

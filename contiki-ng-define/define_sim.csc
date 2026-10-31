<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <project EXPORT="discard">[APPS_DIR]/mrm</project>
  <project EXPORT="discard">[APPS_DIR]/mspsim</project>
  <project EXPORT="discard">[APPS_DIR]/avrora</project>
  <project EXPORT="discard">[APPS_DIR]/serial_socket</project>
  <project EXPORT="discard">[APPS_DIR]/collect-view</project>
  <project EXPORT="discard">[APPS_DIR]/powertracker</project>
  <simulation>
    <title>6TiSCH DEFINE Cooja Simulation</title>
    <speedlimit>1.0</speedlimit>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>50.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>
    <motetype>
      org.contikios.cooja.msp430.Z1MoteType
      <identifier>z1-node</identifier>
      <description>Z1 Node (DEFINE)</description>
      <sourceEX>node.c</sourceEX>
      <commands>make node.z1 TARGET=z1</commands>
      <firmware>[CONTIKI_DIR]/examples/ipv6/rpl-tscha/node.z1</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteSerial</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.BaseRadioCI</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.msp430.interfaces.Z1Flash</moteinterface>
      <moteinterface>org.contikios.cooja.msp430.interfaces.Z1ByteRadio</moteinterface>
      <moteinterface>org.contikios.cooja.msp430.interfaces.Z1Clock</moteinterface>
      <moteinterface>org.contikios.cooja.msp430.interfaces.Z1LED</moteinterface>
      <moteinterface>org.contikios.cooja.msp430.interfaces.Z1Button</moteinterface>
    </motetype>
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.SimControl
    <width>280</width>
    <z>1</z>
    <height>160</height>
    <location_x>400</location_x>
    <location_y>0</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script>
        TIMEOUT(3600000, log.log("Simulation finished successfully.\n"));
        while(true) {
          YIELD();
          log.log(time + ":" + id + ":" + msg + "\n");
        }
      </script>
      <active>true</active>
    </plugin_config>
    <width>600</width>
    <z>0</z>
    <height>700</height>
    <location_x>0</location_x>
    <location_y>0</location_y>
  </plugin>
</simconf>
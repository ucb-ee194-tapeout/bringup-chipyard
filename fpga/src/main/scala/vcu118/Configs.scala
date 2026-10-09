package chipyard.fpga.vcu118

import sys.process._

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{SystemBusKey, PeripheryBusKey, ControlBusKey, ExtMem}
import freechips.rocketchip.devices.debug.{DebugModuleKey, ExportDebug, JTAG}
import freechips.rocketchip.devices.tilelink.{DevNullParams, BootROMLocated}
import freechips.rocketchip.diplomacy.{RegionType, AddressSet}
import freechips.rocketchip.resources.{DTSModel, DTSTimebase}
import freechips.rocketchip.util.{SystemFileName}

import sifive.blocks.devices.spi.{PeripherySPIKey, SPIParams}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}

import sifive.fpgashells.shell.{DesignKey}
import sifive.fpgashells.shell.xilinx.{VCU118ShellPMOD, VCU118DDRSize}

import testchipip.serdes.{SerialTLKey}

import chipyard._
import chipyard.harness._

class WithDefaultPeripherals extends Config((site, here, up) => {
  case PeripheryUARTKey => List(UARTParams(address = BigInt(0x64000000L)))
  case PeripherySPIKey => List(SPIParams(rAddress = BigInt(0x64001000L)))
  case VCU118ShellPMOD => "SDIO"
})

class WithSystemModifications extends Config((site, here, up) => {
  case DTSTimebase => BigInt((1e6).toLong)
  // case BootROMLocated(x) => up(BootROMLocated(x), site).map { p =>
  //   // invoke makefile for sdboot
  //   val freqMHz = (site(SystemBusKey).dtsFrequency.get / (1000 * 1000)).toLong
  //   val make = s"make -C fpga/src/main/resources/vcu118/sdboot PBUS_CLK=${freqMHz} bin"
  //   require (make.! == 0, "Failed to build bootrom")
  //   p.copy(hang = 0x10000, contentFileName = SystemFileName(s"./fpga/src/main/resources/vcu118/sdboot/build/sdboot.bin"))
  // }
  case ExtMem => up(ExtMem, site).map(x => x.copy(master = x.master.copy(size = site(VCU118DDRSize)))) // set extmem to DDR size
  case SerialTLKey => Nil // remove serialized tl port
})

// DOC include start: AbstractVCU118 and Rocket
class WithVCU118Tweaks extends Config(
  // clocking
  new chipyard.harness.WithAllClocksFromHarnessClockInstantiator ++
  new chipyard.clocking.WithPassthroughClockGenerator ++
  new chipyard.config.WithUniformBusFrequencies(100) ++
  new WithFPGAFrequency(100) ++ // default 100MHz freq
  // harness binders
  new WithVCU118UARTTSI ++ // UART-TSI on the on-board USB-UART (default debug/loading path)
  new WithSPISDCard ++
  new WithDDRMem ++
  new WithJTAG ++
  // other configuration
  new chipyard.config.WithNoUART ++ // no peripheral UART by default; opt-in per config
  new WithDefaultPeripherals ++
  new testchipip.tsi.WithUARTTSIClient ++ // enable UART-TSI client (default 115200 baud)
  new chipyard.config.WithTLBackingMemory ++ // use TL backing memory
  new WithSystemModifications ++ // setup busses, use sdboot bootrom, setup ext. mem. size
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNMemoryChannels(1)
)

// Re-enable the peripheral UART (disabled in the shared tweaks) and route it to PMOD J52.
class WithVCU118PeripheralUARTOnPMOD extends Config(
  new WithVCU118UARTToPMOD() ++
  new Config((site, here, up) => {
    case PeripheryUARTKey => List(UARTParams(address = BigInt(0x64000000L)))
  })
)

class RocketVCU118Config extends Config(
  new WithVCU118Tweaks ++
  new chipyard.RocketConfig
)
// DOC include end: AbstractVCU118 and Rocket

class BoomVCU118Config extends Config(
  new WithFPGAFrequency(50) ++
  new WithVCU118Tweaks ++
  new chipyard.MegaBoomV3Config
)

class EE290VCU118Config extends Config(
  new WithoutSDIO ++ // drop SDIO peripheral and free AV15 for UART TX
  new WithVCU118PeripheralUARTOnPMOD ++ // peripheral UART on PMOD J52 (AT15/AT16)
  new WithFPGAFrequency(50) ++
  new WithVCU118Tweaks ++
  new EE290SimConfig
)

// Drop the SPI/SDIO peripheral and switch the PMOD selector to "JTAG" so the
// JTAG debug overlay places itself on PMOD_J52 (with proper GCIO + dedicated-route override)
// instead of FMC_J2. Pairs with the conditional SPI placement in TestHarness.scala
// and with the UART-on-PMOD binder using non-overlapping PMOD pins (AT16, AV15).
class WithoutSDIO extends Config((site, here, up) => {
  case PeripherySPIKey => Nil
  case VCU118ShellPMOD => "JTAG"
})

class SmallBroadcastVCU118Config extends Config(
  new WithVCU118PeripheralUARTOnPMOD ++              // peripheral UART on PMOD J52 (AT16=rx, AV15=tx)
  new WithoutSDIO ++                                 // free AV15 (was SDIO spi_clk)
  new WithFPGAFrequency(50) ++
  new WithVCU118Tweaks ++
  new chipyard.config.WithBroadcastManager ++        // BroadcastHub instead of L2 inclusive cache
  new freechips.rocketchip.rocket.WithNSmallCores(1) ++ // single small Rocket (no FPU, no VM)
  new chipyard.config.AbstractConfig
)

class WithFPGAFrequency(fMHz: Double) extends Config(
  new chipyard.harness.WithHarnessBinderClockFreqMHz(fMHz) ++
  new chipyard.config.WithSystemBusFrequency(fMHz) ++
  new chipyard.config.WithPeripheryBusFrequency(fMHz) ++
  new chipyard.config.WithControlBusFrequency(fMHz) ++
  new chipyard.config.WithFrontBusFrequency(fMHz) ++
  new chipyard.config.WithMemoryBusFrequency(fMHz)
)

class WithFPGAFreq25MHz extends WithFPGAFrequency(25)
class WithFPGAFreq50MHz extends WithFPGAFrequency(50)
class WithFPGAFreq75MHz extends WithFPGAFrequency(75)
class WithFPGAFreq100MHz extends WithFPGAFrequency(100)

// See LICENSE for license details.
package chipyard.fpga.sophialake

import chisel3._

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy._
import org.chipsalliance.diplomacy.lazymodule._

import freechips.rocketchip.diplomacy._
import freechips.rocketchip.devices.debug._
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.system._
import freechips.rocketchip.tile._

import sifive.blocks.devices.uart._
import sifive.fpgashells.shell.{DesignKey}

import testchipip.serdes._
import testchipip.soc.{OBUS}

import chipyard.{BuildSystem}
import chipyard.clocking.{ClockDividerN}
import chipyard.harness.{HarnessClockInstantiator, HarnessClockInstantiatorKey}


//==========================================================
// Serial-TL link definitions
//==========================================================

object SophiaLakeSerialTL {
  /** Port 0 of every bringup design: the decoupled link to the testchip (8-bit by default; FA26 overrides to 32-bit).
    *
    * The FPGA is the manager for the chip's [0, 2 GiB) -- everything below
    * DRAM_BASE, including its CLINT at 0x0200_0000 -- and the chip is a client
    * that may master into this FPGA's memory (DRAM, scratchpad). The bringup
    * platform sources the link clock.
    *
    * `slaveWhere` picks which bus the manager hangs off: OBUS for the
    * single-chip configs, SBUS for the C2C configs (which keep [0, 4 GiB) on
    * this FPGA and reserve the OBUS for the peer).
    */
  def chipLink(slaveWhere: TLBusWrapperLocation = OBUS, phitWidth: Int = 8, flitWidth: Int = 16) = SerialTLParams(
    manager = Some(SerialTLManagerParams(
      memParams = Seq(ManagerRAMParams(
        address = BigInt("00000000", 16),
        size    = BigInt("80000000", 16)
      )),
      slaveWhere = slaveWhere
    )),
    client    = Some(SerialTLClientParams()),
    phyParams = DecoupledInternalSyncSerialPhyParams(phitWidth = phitWidth, flitWidth = flitWidth, freqMHz = 50)
  )

  /** Port 1 of the C2C designs: the 1-bit credited source-synchronous link to
    * the peer FPGA, over 8 PMOD pins.
    *
    * The manager on the OBUS covers the peer's whole 32-bit space, reached
    * through the replicated [4 GiB, 8 GiB) alias (see `SophiaLakeConfig`'s
    * `offchipReplicationBase`): the sender strips the offset, so the wire
    * carries the peer's real low address and everything on the peer is reached
    * at (0x1_0000_0000 + addr):
    *   0x1_0200_0000 -> peer silicon CLINT      (wire 0x0200_0000 -> peer port 0 -> silicon)
    *   0x1_c000_0000 -> peer's local scratchpad (wire 0xc000_0000 -> peer MBUS scratchpad)
    * The client forwards the peer's inbound low-address traffic to our own
    * silicon (port 0) or our own scratchpad (MBUS).
    */
  val peerLink = SerialTLParams(
    manager = Some(SerialTLManagerParams(
      memParams = Seq(ManagerRAMParams(
        address = BigInt("00000000", 16),
        size    = BigInt("100000000", 16)
      )),
      slaveWhere = OBUS
    )),
    client    = Some(SerialTLClientParams()),
    phyParams = CreditedSourceSyncSerialPhyParams(phitWidth = 1, flitWidth = 16, freqMHz = 5)
  )
}

/** Single serial-TL link to the testchip (single-chip bringup). */
class WithSophiaLakeChipSerialTL extends Config(
  new WithSerialTL(Seq(SophiaLakeSerialTL.chipLink())))

/** FA26: 32 lanes per direction with 32-bit flits  */
class WithSophiaLakeFA26ChipSerialTL extends Config(
  new WithSerialTL(Seq(SophiaLakeSerialTL.chipLink(phitWidth = 32, flitWidth = 32))))

/** Port 0 to our own testchip, port 1 to the peer FPGA (C2C bringup). */
class WithSophiaLakeC2CSerialTL extends Config(
  new WithSerialTL(Seq(SophiaLakeSerialTL.chipLink(SBUS), SophiaLakeSerialTL.peerLink)))


//==========================================================
// Bringup configs
//==========================================================

/** Everything a C2C bringup design needs except its board-specific pin binder:
  * both serial-TL ports, a local scratchpad the peer can reach at
  * (0x1_0000_0000 + `scratchpadBase`), a divisible harness clock (the C2C link
  * runs at 5 MHz off the 50 MHz reference), and an address map that keeps
  * [0, 4 GiB) on this FPGA and replicates the peer above it.
  */
class WithSophiaLakeC2C(scratchpadBase: BigInt) extends Config(
  new WithSophiaLakeC2CSerialTL ++
  new testchipip.soc.WithScratchpad(base = scratchpadBase, size = 16 << 10, busWhere = MBUS) ++
  new WithDividerHarnessClockInstantiator ++
  new SophiaLakeConfig(
    freqMHz                = 50,
    offchipBlockRange      = AddressSet.misaligned(0, BigInt(1) << 32),
    offchipReplicationBase = Some(BigInt(1) << 32)))


// ---------------------------------------------------------
// DSP25
// ---------------------------------------------------------
class SophiaLakeDSP25Config extends Config(
  new WithDSP25SophiaLakeSerialTLToGPIO ++
  new WithSophiaLakeChipSerialTL ++
  new SophiaLakeConfig(freqMHz = 50))
  
class SophiaLakeDSP25C2CConfig extends Config(
  new WithDSP25C2CSophiaLakeSerialTLToGPIO ++
  new WithSophiaLakeC2C(scratchpadBase = BigInt("c0000000", 16)))



// ---------------------------------------------------------
// FA26
// ---------------------------------------------------------

class SophiaLakeFA26Config extends Config(
  new WithFA26SophiaLakeSerialTLToGPIO ++
  new WithSophiaLakeFA26ChipSerialTL ++
  new SophiaLakeConfig(freqMHz = 50))


// ---------------------------------------------------------
// BearlyML 25
// ---------------------------------------------------------
class SophiaLakeBML25Config extends Config(
  new WithBML25SophiaLakeSerialTLToGPIO ++
  new WithSophiaLakeChipSerialTL ++
  new SophiaLakeConfig(freqMHz = 50))

class SophiaLakeBML25C2CConfig extends Config(
  new WithBML25C2CSophiaLakeSerialTLToGPIO ++
  new WithSophiaLakeC2C(scratchpadBase = BigInt("d0000000", 16)))


//==========================================================
// Soft-core configs (no testchip -- the FPGA runs the core)
//==========================================================

/** A single Rocket core implemented in the FPGA fabric, booting out of the
  * on-board DDR. Programs and console I/O go over the FTDI USB-UART (T21/U21)
  * with TSI, so the peripheral UART and the JTAG DTM are dropped and no
  * serial-TL link is built. */
class RocketSophiaLakeConfig extends Config(
  new WithSophiaLakeUARTTSI ++                                          // TSI over the FTDI UART pins
  new testchipip.tsi.WithUARTTSIClient(initBaudRate = BigInt(921600)) ++
  new testchipip.serdes.WithNoSerialTL ++                               // no testchip to talk to
  new chipyard.config.WithNoUART ++                                     // the UART pins carry UART-TSI
  new chipyard.config.WithNoDebug ++                                    // no JTAG pins on this board
  new WithSophiaLakeTweaks(freqMHz = 50) ++
  new chipyard.RocketConfig)


//==========================================================
// Generic config classes
//==========================================================

/** Don't use FPGAShell's DesignKey. */
class WithNoDesignKey extends Config((site, here, up) => {
  case DesignKey => (p: Parameters) => new SimpleLazyRawModule()(p)
})

/** Disable the UART-TSI bringup client. */
class WithNoUARTTSI extends Config((site, here, up) => {
  case testchipip.tsi.UARTTSIClientKey => None
})

/** Board-level setup shared by every SophiaLake design: the FTDI UART-TSI
  * bringup link, DDR through the shell's MIG, and a single uniform clock
  * domain at `freqMHz`. */
class WithSophiaLakeTweaks(freqMHz: Double) extends Config(
  new WithNoDesignKey ++
  new WithSophiaLakeUARTTSI ++
  new testchipip.tsi.WithUARTTSIClient(initBaudRate = BigInt(921600)) ++
  new WithSophiaLakeTDDRTL ++
  new chipyard.config.WithBroadcastManager ++                           // no L2
  new chipyard.harness.WithSerialTLTiedOff ++
  new chipyard.harness.WithHarnessBinderClockFreqMHz(freqMHz) ++
  new chipyard.config.WithUniformBusFrequencies(freqMHz) ++
  new chipyard.harness.WithAllClocksFromHarnessClockInstantiator ++
  new chipyard.clocking.WithPassthroughClockGenerator ++
  new chipyard.config.WithTLBackingMemory ++                            // FPGA-shells converts the AXI to TL for us
  new freechips.rocketchip.subsystem.WithExtMemSize(BigInt(256) << 22) ++ // 1 GB of DDR
  new freechips.rocketchip.subsystem.WithoutTLMonitors)

/** The coreless bringup host: an offchip bus hanging off the SBUS, carrying the
  * serial-TL link to the testchip.
  *
  * The C2C configs override the address-map arguments to replicate the peer's
  * space above `offchipReplicationBase`; the defaults keep the original
  * single-chip bringup topology, with the testchip's main memory off the OBUS.
  */
class SophiaLakeConfig(
  freqMHz: Double,
  offchipBlockRange: Seq[AddressSet] = AddressSet.misaligned(0x80000000L, (BigInt(1) << 30) * 4),
  offchipReplicationBase: Option[BigInt] = None
) extends Config(
  new testchipip.soc.WithOffchipBusClient(SBUS,
    blockRange      = offchipBlockRange,
    replicationBase = offchipReplicationBase) ++
  new testchipip.soc.WithOffchipBus ++
  new WithSophiaLakeTweaks(freqMHz = freqMHz) ++
  new chipyard.NoCoresConfig)


class DividerHarnessClockInstantiator extends HarnessClockInstantiator {
  def instantiateHarnessClocks(refClock: Clock, refClockFreqMHz: Double): Unit = {
    val refFreqHz = refClockFreqMHz * 1000 * 1000
    for ((name, (freqHz, clock)) <- clockMap) {
      if (freqHz == refFreqHz) {
        clock := refClock
      } else {
        val divBy = math.round(refFreqHz / freqHz).toInt
        require(divBy > 1 && math.abs(refFreqHz / divBy - freqHz) < 1.0,
          s"Reference clock ${refClockFreqMHz} MHz cannot be evenly divided to get ${freqHz / 1e6} MHz for clock $name")
        val divider = Module(new ClockDividerN(divBy))
        divider.io.clk_in := refClock
        clock := divider.io.clk_out
      }
    }
  }
}

class WithDividerHarnessClockInstantiator extends Config((site, here, up) => {
  case HarnessClockInstantiatorKey => () => new DividerHarnessClockInstantiator
})

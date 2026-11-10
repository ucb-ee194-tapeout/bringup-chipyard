package chipyard.fpga.sophialake

import chisel3._

import freechips.rocketchip.diplomacy.{LazyRawModuleImp}
import org.chipsalliance.diplomacy.nodes.{HeterogeneousBag}

import sifive.blocks.devices.uart.{UARTPortIO}
import sifive.fpgashells.shell._
import sifive.fpgashells.ip.xilinx._
import sifive.fpgashells.shell.xilinx._
import sifive.fpgashells.clocks._

import chipyard._
import chipyard.harness._
import chipyard.iobinders._
import testchipip.serdes._


//==========================================================
// Pin maps and constraint helpers
//==========================================================

/** Package pins for one decoupled serial-TL port, in bit order. */
case class DecoupledSerialTLPins(
  clock:    String,
  outValid: String,
  outReady: String,
  outPhit:  Seq[String],
  inValid:  String,
  inReady:  String,
  inPhit:   Seq[String])

/** Package pins for one 1-bit credited source-synchronous C2C port. The two
  * boards use mirrored maps, since the PMOD cable runs straight through. */
case class CreditedC2CPins(
  clockOut:  String,
  resetOut:  String,
  outValid:  String,
  outPhit:   String,
  clockIn:   String,
  resetIn:   String,
  inValid:   String,
  inPhit:    String)

object SophiaLakePins {
  // TL_IN goes to FPGA out; TL_OUT comes from the chip to FPGA in.
  val fa26ChipLink = DecoupledSerialTLPins(
    clock    = "B20",
    outValid = "B22", outReady = "L18",
    outPhit  = Seq(
      "B15", "L19", "G18", "K18", "F16", "L20", "G17", "M20",
      "F14", "H22", "G16", "J21", "K16", "J22", "L16", "H19",
      "M15", "C14", "A14", "C15", "C13", "M18", "J14", "M22",
      "D14", "D17", "G13", "K22", "B13", "B16", "A15", "C18"),
    inValid  = "B17", inReady  = "J17",
    inPhit   = Seq(
      "C22", "K17", "E22", "G15", "E19", "H15", "F19", "H14",
      "G20", "J15", "G21", "N20", "H20", "L15", "C19", "N18",
      "D21", "A13", "N22", "N19", "K21", "L14", "D16", "K13",
      "M21", "M16", "L21", "A16", "C17", "H13", "F18", "E13"))

  val dsp25ChipLink = DecoupledSerialTLPins(
    clock    = "A21",
    outValid = "B20", outReady = "E16",
    outPhit  = Seq("C22", "E19", "M15", "A15", "C13", "L15", "G20", "F13"),
    inValid  = "G13", inReady  = "E17",
    inPhit   = Seq("G15", "H14", "J20", "J17", "N20", "N19", "J19", "A14"))

  val bml25ChipLink = DecoupledSerialTLPins(
    clock    = "D17",
    outValid = "C15", outReady = "D16",
    outPhit  = Seq("C17", "A18", "B21", "A20", "B18", "A19", "C18", "B17"),
    inValid  = "A21", inReady  = "B20",
    inPhit   = Seq("H19", "G20", "C14", "H20", "F19", "E22", "C19", "C22"))

  /** DSP25 drives PMOD 00-01 and listens on PMOD 02-03. */
  val dsp25C2C = CreditedC2CPins(
    clockOut = "Y16",  resetOut = "AA16", outValid = "AB16", outPhit = "AB17",
    clockIn  = "AA13", resetIn  = "AB13", inValid  = "AA15", inPhit  = "AB15")

  /** BML25 is the mirror image of DSP25: it drives what DSP25 listens on. */
  val bml25C2C = CreditedC2CPins(
    clockOut = "AB15", resetOut = "AA15", outValid = "AB13", outPhit = "AA13",
    clockIn  = "AB17", resetIn  = "AB16", inValid  = "AA16", inPhit  = "Y16")

  /** Constrain each (package pin, IO) pair to `ioStandard`. */
  def addPins(
    ath: SophiaLakeHarness,
    pins: Seq[(String, IOPin)],
    ioStandard: String,
    iob: Boolean = true,
    pullup: Boolean = false
  ): Unit = pins.foreach { case (pin, io) =>
    ath.xdc.addPackagePin(io, pin)
    ath.xdc.addIOStandard(io, ioStandard)
    if (iob)    ath.xdc.addIOB(io)
    if (pullup) ath.xdc.addPullup(io)
  }

  /** Pin out a decoupled serial-TL port and constrain its forwarded clock. */
  def constrainChipLink(ath: SophiaLakeHarness, io: DecoupledPhitIO, pins: DecoupledSerialTLPins): Unit = {
    val clkIO = io match {
      case io: HasClockOut => IOPin(io.clock_out)
      case io: HasClockIn  => IOPin(io.clock_in)
    }
    val packagePinsWithPackageIOs =
      Seq(
        (pins.clock,    clkIO),
        (pins.outValid, IOPin(io.out.valid)),
        (pins.outReady, IOPin(io.out.ready))
      ) ++
      pins.outPhit.zipWithIndex.map { case (pin, i) => (pin, IOPin(io.out.bits.phit, i)) } ++
      Seq(
        (pins.inValid, IOPin(io.in.valid)),
        (pins.inReady, IOPin(io.in.ready))
      ) ++
      pins.inPhit.zipWithIndex.map { case (pin, i) => (pin, IOPin(io.in.bits.phit, i)) }

    addPins(ath, packagePinsWithPackageIOs, "LVCMOS12", iob = false)

    // Don't add IOB to the clock, if its an input
    val iobPins = io match {
      case _: DecoupledInternalSyncPhitIO => packagePinsWithPackageIOs
      case _: DecoupledExternalSyncPhitIO => packagePinsWithPackageIOs.drop(1)
    }
    iobPins.foreach { case (_, io) => ath.xdc.addIOB(io) }

    ath.sdc.addClock("ser_tl_clock", clkIO, 100)
    ath.sdc.addGroup(pins = Seq(clkIO))
    ath.xdc.clockDedicatedRouteFalse(clkIO)
  }

  //==========================================================
  // FA26 Sophia Lake: 1.8 V chip-link IO
  //==========================================================

  /** Dedicated FA26 constraints, kept separate from the legacy chip links. */
  def constrainFA26ChipLink(ath: SophiaLakeHarness, io: DecoupledPhitIO): Unit = {
    val pins = fa26ChipLink
    val clkIO = io match {
      case io: HasClockOut => IOPin(io.clock_out)
      case io: HasClockIn  => IOPin(io.clock_in)
    }
    val packagePinsWithPackageIOs =
      Seq(
        (pins.clock,    clkIO),
        (pins.outValid, IOPin(io.out.valid)),
        (pins.outReady, IOPin(io.out.ready))
      ) ++
      pins.outPhit.zipWithIndex.map { case (pin, i) => (pin, IOPin(io.out.bits.phit, i)) } ++
      Seq(
        (pins.inValid, IOPin(io.in.valid)),
        (pins.inReady, IOPin(io.in.ready))
      ) ++
      pins.inPhit.zipWithIndex.map { case (pin, i) => (pin, IOPin(io.in.bits.phit, i)) }

    addPins(ath, packagePinsWithPackageIOs, "LVCMOS18", iob = false)

    // Don't add IOB to the clock, if its an input
    val iobPins = io match {
      case _: DecoupledInternalSyncPhitIO => packagePinsWithPackageIOs
      case _: DecoupledExternalSyncPhitIO => packagePinsWithPackageIOs.drop(1)
    }
    iobPins.foreach { case (_, io) => ath.xdc.addIOB(io) }

    ath.sdc.addClock("ser_tl_clock", clkIO, 100)
    ath.sdc.addGroup(pins = Seq(clkIO))
    ath.xdc.clockDedicatedRouteFalse(clkIO)
  }

  /** Pin out a credited source-sync C2C port.
    *
    * The forwarded clock goes out INVERTED (180 deg) so the far side samples in
    * the middle of the data eye rather than on the launching edge, which buys
    * setup/hold margin on the slow PMOD link. Only the outgoing clock is
    * flipped; clock_in is left untouched (the peer inverts its own).
    */
  def constrainC2CLink(
    ath: SophiaLakeHarness,
    io: CreditedSourceSyncPhitIO,
    dutIO: CreditedSourceSyncPhitIO,
    pins: CreditedC2CPins
  ): Unit = {
    io.clock_out := (~dutIO.clock_out.asUInt).asBool.asClock

    addPins(ath, Seq(
      (pins.clockOut, IOPin(io.clock_out)),
      (pins.resetOut, IOPin(io.reset_out)),
      (pins.outValid, IOPin(io.out.valid)),
      (pins.outPhit,  IOPin(io.out.bits.phit, 0)),
      (pins.clockIn,  IOPin(io.clock_in)),
      (pins.resetIn,  IOPin(io.reset_in)),
      (pins.inValid,  IOPin(io.in.valid)),
      (pins.inPhit,   IOPin(io.in.bits.phit, 0))
    ), "LVCMOS33")

    ath.sdc.addClock("c2c_clock_in", IOPin(io.clock_in), 100)
    ath.sdc.addGroup(pins = Seq(IOPin(io.clock_in)))
    ath.xdc.clockDedicatedRouteFalse(IOPin(io.clock_in))
  }

  def harness(th: HasHarnessInstantiators): SophiaLakeHarness =
    th.asInstanceOf[LazyRawModuleImp].wrapper.asInstanceOf[SophiaLakeHarness]
}


//==========================================================
// Serial-TL to GPIO binders
//==========================================================

/** Copy the configured 32-bit IO and reuse the board's constraint helper. */
class WithFA26SophiaLakeSerialTLToGPIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: SerialTLPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName("serial_tl")
    harnessIO <> port.io
    harnessIO match {
      case io: DecoupledPhitIO =>
        SophiaLakePins.constrainFA26ChipLink(ath, io)
    }
  }
})

class WithDSP25SophiaLakeSerialTLToGPIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: SerialTLPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName("serial_tl")
    harnessIO <> port.io
    harnessIO match {
      case io: DecoupledPhitIO => SophiaLakePins.constrainChipLink(ath, io, SophiaLakePins.dsp25ChipLink)
    }
  }
})


class WithBML25SophiaLakeSerialTLToGPIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: SerialTLPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName("serial_tl")
    harnessIO <> port.io
    harnessIO match {
      case io: DecoupledPhitIO => SophiaLakePins.constrainChipLink(ath, io, SophiaLakePins.bml25ChipLink)
    }
  }
})


/** C2C designs have two ports: port 0 is the decoupled link to our own chip
  * (same pins as the single-chip config), port 1 the credited source-sync link
  * to the peer FPGA over the PMOD. */
class WithDSP25C2CSophiaLakeSerialTLToGPIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: SerialTLPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName(s"serial_tl_${port.portId}")
    harnessIO <> port.io
    harnessIO match {
      case io: DecoupledPhitIO if port.portId == 0 =>
        SophiaLakePins.constrainChipLink(ath, io, SophiaLakePins.dsp25ChipLink)
      case io: CreditedSourceSyncPhitIO if port.portId == 1 =>
        SophiaLakePins.constrainC2CLink(ath, io,
          port.io.asInstanceOf[CreditedSourceSyncPhitIO], SophiaLakePins.dsp25C2C)
    }
  }
})


class WithBML25C2CSophiaLakeSerialTLToGPIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: SerialTLPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName(s"serial_tl_${port.portId}")
    harnessIO <> port.io
    harnessIO match {
      case io: DecoupledPhitIO if port.portId == 0 =>
        SophiaLakePins.constrainChipLink(ath, io, SophiaLakePins.bml25ChipLink)
      case io: CreditedSourceSyncPhitIO if port.portId == 1 =>
        SophiaLakePins.constrainC2CLink(ath, io,
          port.io.asInstanceOf[CreditedSourceSyncPhitIO], SophiaLakePins.bml25C2C)
    }
  }
})


//==========================================================
// Bringup link binders
//==========================================================

/** TSI over the on-board FTDI USB-UART. */
class WithSophiaLakeUARTTSI extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: UARTTSIPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val harnessIO = IO(new UARTPortIO(port.io.uartParams)).suggestName("uart_tsi")
    harnessIO <> port.io.uart

    SophiaLakePins.addPins(ath, Seq(
      ("T21", IOPin(harnessIO.rxd)),
      ("U21", IOPin(harnessIO.txd))
    ), "LVCMOS33")
  }
})


//==========================================================
// Peripheral binders
//==========================================================

/** Route the design's TileLink memory port to the shell's DDR client. */
class WithSophiaLakeTDDRTL extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: TLMemPort, chipId: Int) => {
    val ath = SophiaLakePins.harness(th)
    val bundles = ath.ddrClient.out.map(_._1)
    val ddrClientBundle = Wire(new HeterogeneousBag(bundles.map(_.cloneType)))
    bundles.zip(ddrClientBundle).foreach { case (bundle, io) => bundle <> io }
    ddrClientBundle <> port.io
  }
})

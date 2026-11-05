package chipyard.fpga.vcu118

import chisel3._
import chisel3.experimental.{BaseModule}

import org.chipsalliance.diplomacy.nodes.{HeterogeneousBag}
import freechips.rocketchip.tilelink.{TLBundle}

import sifive.blocks.devices.uart.{UARTPortIO}
import sifive.blocks.devices.spi.{HasPeripherySPI, SPIPortIO}

import sifive.fpgashells.shell.IOPin

import chipyard._
import chipyard.harness._
import chipyard.iobinders._

/*** UART ***/
class WithUART extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: UARTPort, chipId: Int) => {
    th.vcu118Outer.io_uart_bb.bundle <> port.io
  }
})

// UART-TSI on the on-board USB-UART (via the VCU118 UART overlay)
class WithVCU118UARTTSI extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: UARTTSIPort, chipId: Int) => {
    th.vcu118Outer.io_uart_bb.bundle <> port.io.uart
  }
})



class WithVCU118UARTToPMOD(
  ioStandard: String = "LVCMOS18"
) extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: UARTPort, chipId: Int) => {
    val vth = th.vcu118Outer
    val harnessIO = IO(chiselTypeOf(port.io)).suggestName("uart_periph")
    harnessIO <> port.io

    val rxdIO = IOPin(harnessIO.rxd) // chip's UART RX  (driven by FTDI TX)
    val txdIO = IOPin(harnessIO.txd) // chip's UART TX  (drives FTDI RX)

    val packagePinsWithPackageIOs = Seq(
      ("AT16", rxdIO), // chip RX -- PMOD J52 pin 10 (FTDI TX wires here)
      ("AV15", txdIO), // chip TX -- PMOD J52 pin 9  (was SDIO spi_clk; freed by dropping SDIO)
    )

    packagePinsWithPackageIOs.foreach { case (pkgPin, io) =>
      vth.xdc.addPackagePin(io, pkgPin)
      vth.xdc.addIOStandard(io, ioStandard)
      vth.xdc.addIOB(io)
    }
    // Pullup on RX so an unconnected line idles high instead of detecting a spurious start bit.
    vth.xdc.addPullup(rxdIO)
  }
})

/*** SPI ***/
class WithSPISDCard extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: SPIPort, chipId: Int) => {
    th.vcu118Outer.io_spi_bb.get.bundle <> port.io
  }
})

/*** Experimental DDR ***/
class WithDDRMem extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: TLMemPort, chipId: Int) => {
    val bundles = th.vcu118Outer.ddrClient.out.map(_._1)
    val ddrClientBundle = Wire(new HeterogeneousBag(bundles.map(_.cloneType)))
    bundles.zip(ddrClientBundle).foreach { case (bundle, io) => bundle <> io }
    ddrClientBundle <> port.io
  }
})

class WithJTAG extends HarnessBinder({
  case (th: VCU118FPGATestHarnessImp, port: JTAGPort, chipId: Int) => {
    val jtag_io = th.vcu118Outer.jtagPlacedOverlay.overlayOutput.jtag.getWrappedValue
    port.io.TCK := jtag_io.TCK
    port.io.TMS := jtag_io.TMS
    port.io.TDI := jtag_io.TDI
    port.io.reset.foreach(_ := th.referenceReset)
    jtag_io.TDO.data := port.io.TDO
    jtag_io.TDO.driven := true.B
    // ignore srst_n
    jtag_io.srst_n := DontCare

  }
})

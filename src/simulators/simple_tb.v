`timescale 1ns/1ps

module simple_tb;

// Clock and reset
reg clk;
reg rst_n;

// TLP Interface
reg         tlp_valid;
reg  [2:0]  tlp_type;
reg  [31:0] tlp_address;
reg  [31:0] tlp_data;
reg  [7:0]  tlp_tag;
reg  [9:0]  tlp_length;
wire        tlp_ready;

// Completion Interface
wire        cpl_valid;
wire [2:0]  cpl_status;
wire [31:0] cpl_data;
wire [7:0]  cpl_tag;
reg         cpl_ready;

// Error Injection
reg         inject_crc_error;
reg         inject_timeout;
reg         inject_ecrc_error;
reg         inject_malformed_tlp;

// Error Status
wire        error_valid;
wire [3:0]  error_type;
wire [63:0] error_header;

// Link Status
wire [3:0]  ltssm_state;
wire        link_up;
wire [2:0]  link_speed;
wire [4:0]  link_width;

// Instantiate DUT
pcie_lite dut (
    .clk(clk),
    .rst_n(rst_n),
    .tlp_valid(tlp_valid),
    .tlp_type(tlp_type),
    .tlp_address(tlp_address),
    .tlp_data(tlp_data),
    .tlp_tag(tlp_tag),
    .tlp_length(tlp_length),
    .tlp_ready(tlp_ready),
    .cpl_valid(cpl_valid),
    .cpl_status(cpl_status),
    .cpl_data(cpl_data),
    .cpl_tag(cpl_tag),
    .cpl_ready(cpl_ready),
    .inject_crc_error(inject_crc_error),
    .inject_timeout(inject_timeout),
    .inject_ecrc_error(inject_ecrc_error),
    .inject_malformed_tlp(inject_malformed_tlp),
    .error_valid(error_valid),
    .error_type(error_type),
    .error_header(error_header),
    .ltssm_state(ltssm_state),
    .link_up(link_up),
    .link_speed(link_speed),
    .link_width(link_width)
);

// Clock generation - 100MHz (10ns period)
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// VCD dump
initial begin
    $dumpfile("pcie_waveform.vcd");
    $dumpvars(0, simple_tb);
    $display("VCD dump started - file: pcie_waveform.vcd");
end

// Test stimulus
initial begin
    // Initialize signals
    rst_n = 0;
    tlp_valid = 0;
    tlp_type = 0;
    tlp_address = 0;
    tlp_data = 0;
    tlp_tag = 0;
    tlp_length = 1;
    cpl_ready = 1;
    inject_crc_error = 0;
    inject_timeout = 0;
    inject_ecrc_error = 0;
    inject_malformed_tlp = 0;
    
    // Reset sequence
    #100;
    rst_n = 1;
    #100;
    
    // Wait for link up
    wait(link_up);
    $display("Time %0t: Link is up!", $time);
    
    // Test 1: Normal read transaction
    #50;
    $display("\nTime %0t: Sending normal read request", $time);
    send_tlp(3'b000, 32'h1000, 32'h0, 8'd1);
    
    // Test 2: Normal write transaction
    #200;
    $display("\nTime %0t: Sending write request", $time);
    send_tlp(3'b001, 32'h2000, 32'hDEADBEEF, 8'd2);
    
    // Test 3: CRC error injection
    #200;
    $display("\nTime %0t: Injecting CRC error", $time);
    inject_crc_error = 1;
    #50;
    send_tlp(3'b000, 32'h3000, 32'h0, 8'd3);
    #50;
    inject_crc_error = 0;
    
    // Wait for recovery
    #500;
    
    // Test 4: Timeout error
    #200;
    $display("\nTime %0t: Injecting timeout error", $time);
    inject_timeout = 1;
    send_tlp(3'b000, 32'h4000, 32'h0, 8'd4);
    #100;
    inject_timeout = 0;
    
    // Test 5: ECRC error
    #1000;
    $display("\nTime %0t: Injecting ECRC error", $time);
    inject_ecrc_error = 1;
    #50;
    send_tlp(3'b001, 32'h5000, 32'hCAFEBABE, 8'd5);
    #50;
    inject_ecrc_error = 0;
    
    // Test 6: Malformed TLP
    #200;
    $display("\nTime %0t: Injecting malformed TLP", $time);
    inject_malformed_tlp = 1;
    send_tlp(3'b111, 32'h6000, 32'h0, 8'd6);  // Invalid TLP type
    #50;
    inject_malformed_tlp = 0;
    
    // Final transactions
    #200;
    $display("\nTime %0t: Sending final transactions", $time);
    send_tlp(3'b000, 32'h7000, 32'h0, 8'd7);
    #200;
    send_tlp(3'b001, 32'h8000, 32'h12345678, 8'd8);
    
    // End simulation
    #1000;
    $display("\nSimulation completed at time %0t", $time);
    $display("Total errors detected: Check waveform for error_valid signal");
    $finish;
end

// Task to send TLP
task send_tlp;
    input [2:0] ttype;
    input [31:0] addr;
    input [31:0] data;
    input [7:0] tag;
    begin
        wait(tlp_ready);
        @(posedge clk);
        tlp_valid <= 1;
        tlp_type <= ttype;
        tlp_address <= addr;
        tlp_data <= data;
        tlp_tag <= tag;
        tlp_length <= 1;
        @(posedge clk);
        tlp_valid <= 0;
    end
endtask

// Monitor errors
always @(posedge clk) begin
    if (error_valid) begin
        $display("Time %0t: ERROR detected! Type=%0d", $time, error_type);
    end
end

// Monitor completions
always @(posedge clk) begin
    if (cpl_valid && cpl_ready) begin
        $display("Time %0t: Completion received - Tag=%0d Status=%0d", 
                 $time, cpl_tag, cpl_status);
    end
end

// Monitor LTSSM state changes
reg [3:0] prev_ltssm_state;
always @(posedge clk) begin
    if (prev_ltssm_state != ltssm_state) begin
        $display("Time %0t: LTSSM state change: %0h -> %0h", 
                 $time, prev_ltssm_state, ltssm_state);
        prev_ltssm_state <= ltssm_state;
    end
end

endmodule
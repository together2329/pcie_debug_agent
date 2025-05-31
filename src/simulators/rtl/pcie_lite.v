`timescale 1ns/1ps

module pcie_lite (
    input           clk,
    input           rst_n,
    
    // TLP Interface (simplified)
    input           tlp_valid,
    input   [2:0]   tlp_type,      // 3'b000: MRd, 3'b001: MWr, 3'b010: CplD
    input   [31:0]  tlp_address,
    input   [31:0]  tlp_data,
    input   [7:0]   tlp_tag,
    input   [9:0]   tlp_length,    // DW length
    output reg      tlp_ready,
    
    // Completion Interface
    output reg      cpl_valid,
    output reg [2:0] cpl_status,   // 3'b000: SC, 3'b001: UR, 3'b010: CA
    output reg [31:0] cpl_data,
    output reg [7:0]  cpl_tag,
    input           cpl_ready,
    
    // Error Injection
    input           inject_crc_error,
    input           inject_timeout,
    input           inject_ecrc_error,
    input           inject_malformed_tlp,
    
    // Error Status
    output reg      error_valid,
    output reg [3:0] error_type,
    output reg [63:0] error_header,
    
    // Link Status
    output reg [3:0] ltssm_state,
    output reg       link_up,
    output reg [2:0] link_speed,    // 3'b001: Gen1, 3'b010: Gen2, 3'b011: Gen3
    output reg [4:0] link_width     // 5'b00001: x1, 5'b00100: x4, 5'b10000: x16
);

// LTSSM States
localparam LTSSM_DETECT         = 4'h0;
localparam LTSSM_POLLING        = 4'h1;
localparam LTSSM_CONFIG         = 4'h2;
localparam LTSSM_L0             = 4'h3;
localparam LTSSM_RECOVERY       = 4'h4;
localparam LTSSM_DISABLED       = 4'hF;

// Error Types
localparam ERR_NONE             = 4'h0;
localparam ERR_CRC              = 4'h1;
localparam ERR_TIMEOUT          = 4'h2;
localparam ERR_ECRC             = 4'h3;
localparam ERR_MALFORMED        = 4'h4;
localparam ERR_UNSUPPORTED      = 4'h5;
localparam ERR_POISONED         = 4'h6;

// Internal signals
reg [15:0] timeout_counter;
reg [7:0]  pending_tag;
reg        pending_request;
reg [31:0] mem_array [0:255];  // Simple memory model

// Initialize
initial begin
    ltssm_state = LTSSM_DETECT;
    link_up = 1'b0;
    link_speed = 3'b011;  // Gen3
    link_width = 5'b10000; // x16
end

// LTSSM State Machine
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ltssm_state <= LTSSM_DETECT;
        link_up <= 1'b0;
    end else begin
        case (ltssm_state)
            LTSSM_DETECT: begin
                ltssm_state <= LTSSM_POLLING;
                $display("[%0t] PCIe: LTSSM entering POLLING", $time);
            end
            
            LTSSM_POLLING: begin
                if (inject_timeout) begin
                    ltssm_state <= LTSSM_DETECT;
                    $display("[%0t] PCIe: ERROR - Link training timeout in POLLING", $time);
                end else begin
                    ltssm_state <= LTSSM_CONFIG;
                    $display("[%0t] PCIe: LTSSM entering CONFIG", $time);
                end
            end
            
            LTSSM_CONFIG: begin
                ltssm_state <= LTSSM_L0;
                link_up <= 1'b1;
                $display("[%0t] PCIe: Link UP - Speed Gen%0d Width x%0d", 
                        $time, link_speed, link_width);
            end
            
            LTSSM_L0: begin
                if (inject_crc_error || inject_ecrc_error) begin
                    ltssm_state <= LTSSM_RECOVERY;
                    $display("[%0t] PCIe: ERROR - Entering RECOVERY due to errors", $time);
                end
            end
            
            LTSSM_RECOVERY: begin
                ltssm_state <= LTSSM_L0;
                $display("[%0t] PCIe: RECOVERY complete, back to L0", $time);
            end
        endcase
    end
end

// TLP Processing
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tlp_ready <= 1'b1;
        cpl_valid <= 1'b0;
        pending_request <= 1'b0;
        timeout_counter <= 16'h0;
    end else begin
        // Handle incoming TLPs
        if (tlp_valid && tlp_ready && link_up) begin
            tlp_ready <= 1'b0;
            
            // Log the transaction
            $display("[%0t] PCIe: TLP Received - Type=%0h Addr=0x%08x Data=0x%08x Tag=%0d", 
                    $time, tlp_type, tlp_address, tlp_data, tlp_tag);
            
            // Check for malformed TLP
            if (inject_malformed_tlp) begin
                error_valid <= 1'b1;
                error_type <= ERR_MALFORMED;
                error_header <= {tlp_type, 5'b0, tlp_tag, 16'b0, tlp_address};
                $display("[%0t] PCIe: ERROR - Malformed TLP detected", $time);
            end
            // Process based on TLP type
            else begin
                case (tlp_type)
                    3'b000: begin // Memory Read
                        pending_request <= 1'b1;
                        pending_tag <= tlp_tag;
                        timeout_counter <= 16'hFFFF;
                    end
                    
                    3'b001: begin // Memory Write
                        mem_array[tlp_address[7:0]] <= tlp_data;
                        // No completion needed for writes
                    end
                    
                    default: begin
                        error_valid <= 1'b1;
                        error_type <= ERR_UNSUPPORTED;
                        $display("[%0t] PCIe: ERROR - Unsupported TLP type %0h", 
                                $time, tlp_type);
                    end
                endcase
            end
            
            tlp_ready <= 1'b1;
        end
        
        // Generate completions
        if (pending_request && !cpl_valid) begin
            if (inject_timeout && timeout_counter < 16'h8000) begin
                error_valid <= 1'b1;
                error_type <= ERR_TIMEOUT;
                $display("[%0t] PCIe: ERROR - Completion timeout for tag %0d", 
                        $time, pending_tag);
                pending_request <= 1'b0;
            end else if (timeout_counter > 0) begin
                timeout_counter <= timeout_counter - 1;
                
                // Generate completion after some delay
                if (timeout_counter == 16'hFF00) begin
                    cpl_valid <= 1'b1;
                    cpl_tag <= pending_tag;
                    cpl_data <= mem_array[0]; // Simplified
                    cpl_status <= inject_crc_error ? 3'b010 : 3'b000; // CA or SC
                    pending_request <= 1'b0;
                    
                    $display("[%0t] PCIe: Completion generated - Tag=%0d Status=%0h", 
                            $time, pending_tag, cpl_status);
                end
            end
        end
        
        // Clear completion when accepted
        if (cpl_valid && cpl_ready) begin
            cpl_valid <= 1'b0;
        end
        
        // Error injection
        if (inject_crc_error && !error_valid) begin
            error_valid <= 1'b1;
            error_type <= ERR_CRC;
            $display("[%0t] PCIe: ERROR - CRC error injected", $time);
        end
        
        if (inject_ecrc_error && !error_valid) begin
            error_valid <= 1'b1;
            error_type <= ERR_ECRC;
            $display("[%0t] PCIe: ERROR - ECRC error injected", $time);
        end
        
        // Clear error after one cycle
        if (error_valid) begin
            error_valid <= 1'b0;
        end
    end
end

endmodule
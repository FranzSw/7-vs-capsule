class Config:
    def __init__(self, dc_num_capsules=4, input_width=4, input_height= 256):
       
        ## CNN (cnn)
        #self.cnn_in_channels = 1
        #self.cnn_out_channels = 256
        #self.cnn_kernel_size = 9
        ## Primary Capsule (pc)
        #self.pc_num_capsules = 4
        #self.pc_in_channels = 256
        #self.pc_out_channels = 32
        #self.pc_kernel_size = 9
        #self.pc_num_routes = 32 * 4 * 4
        ## Digit Capsule (dc)
        #self.dc_num_capsules = dc_num_capsules
        #self.dc_num_routes = 32 * 4 * 4
        #self.dc_in_channels = 4
        #self.dc_out_channels = 16
        ## Decoder
        #self.input_width = input_width
        #self.input_height = input_height
        self.cnn_in_channels = 3
        self.cnn_out_channels = 256
        self.cnn_kernel_size = 9
        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 256
        self.pc_out_channels = 32
        self.pc_kernel_size = 9
        self.pc_num_routes = 32 * 8 * 8
        # Digit Capsule (dc)
        self.dc_num_capsules = 4
        self.dc_num_routes = 32 * 8 * 8
        self.dc_in_channels = 8
        self.dc_out_channels = 16
        # Decoder
        self.input_width = 32
        self.input_height = 32  
        
      
    
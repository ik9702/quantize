from torch import nn

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        
        def CBR(in_channels, out_channels):
            cbr = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
            )
            return cbr
        def FIN(in_channels, out_channels):
            fin = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid()
            )
            return fin
        self.MP = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=False)
        
        self.enc1 = CBR(in_channels=1, out_channels=32)
        self.enc2 = CBR(in_channels=32, out_channels=64)
        self.enc3 = CBR(in_channels=64, out_channels=128)
        
        
        self.dec3 = CBR(in_channels=128, out_channels=64)
        self.UP3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = CBR(in_channels=64, out_channels=32)
        self.UP2 = nn.ConvTranspose2d(in_channels=32, out_channels=32,kernel_size=2, stride=2, output_padding=(0,1))
        self.dec1 = CBR(in_channels=32, out_channels=16)
        self.UP1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2,output_padding=(1,1))
        
        
        self.fin = FIN(in_channels=16, out_channels=1)
        
        
    def forward(self, x):
        x = self.MP(self.enc1(x))#129_64
        x = self.MP(self.enc2(x))#64_32
        x = self.MP(self.enc3(x))#32_16
        

        x = self.UP3(self.dec3(x))#16_32
        x = self.UP2(self.dec2(x))#32_64
        x = self.UP1(self.dec1(x))#64_128
        x = self.fin(x)
        return x
            
        
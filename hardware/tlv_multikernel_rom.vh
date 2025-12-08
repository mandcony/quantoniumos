   function automatic logic signed [15:0]
      kernel_real(input logic [3:0] mode, input logic [2:0] k, input logic [2:0] n);
      // MULTI-KERNEL ROM: All 12 proven RFT transforms
      // Generated from algorithms/rft/variants/operator_variants.py
      // December 2025: Updated to match fpga_top.sv
      logic [9:0] idx;
      idx = {mode, k, n};
      unique case (idx)
         // MODE 0: RFT-Golden (unitarity: 6.12e-15)
         10'd  0: kernel_real = 16'shD6E0;  // [0,0]=-0.32130
         10'd  1: kernel_real = 16'sh3209;  // [0,1]=+0.39091
         10'd  2: kernel_real = 16'shD1F4;  // [0,2]=-0.35975
         10'd  3: kernel_real = 16'shCF17;  // [0,3]=-0.38211
         10'd  4: kernel_real = 16'sh36D5;  // [0,4]=+0.42836
         10'd  5: kernel_real = 16'shC836;  // [0,5]=-0.43585
         10'd  6: kernel_real = 16'shDAEF;  // [0,6]=-0.28957
         10'd  7: kernel_real = 16'shF271;  // [0,7]=-0.10592
         10'd  8: kernel_real = 16'shD2A2;  // [1,0]=-0.35442
         10'd  9: kernel_real = 16'sh301E;  // [1,1]=+0.37591
         10'd 10: kernel_real = 16'sh2BF1;  // [1,2]=+0.34328
         10'd 11: kernel_real = 16'shE165;  // [1,3]=-0.23911
         10'd 12: kernel_real = 16'sh2642;  // [1,4]=+0.29888
         10'd 13: kernel_real = 16'sh3DE6;  // [1,5]=+0.48358
         10'd 14: kernel_real = 16'sh3457;  // [1,6]=+0.40892
         10'd 15: kernel_real = 16'sh214C;  // [1,7]=+0.26013
         10'd 16: kernel_real = 16'shD0C9;  // [2,0]=-0.36887
         10'd 17: kernel_real = 16'shD805;  // [2,1]=-0.31233
         10'd 18: kernel_real = 16'sh2C5A;  // [2,2]=+0.34648
         10'd 19: kernel_real = 16'shC4D2;  // [2,3]=-0.46234
         10'd 20: kernel_real = 16'shDDDE;  // [2,4]=-0.26667
         10'd 21: kernel_real = 16'sh1BBC;  // [2,5]=+0.21669
         10'd 22: kernel_real = 16'shCACC;  // [2,6]=-0.41566
         10'd 23: kernel_real = 16'shCFD0;  // [2,7]=-0.37647
         10'd 24: kernel_real = 16'shD0F4;  // [3,0]=-0.36754
         10'd 25: kernel_real = 16'shD5DF;  // [3,1]=-0.32912
         10'd 26: kernel_real = 16'shD160;  // [3,2]=-0.36426
         10'd 27: kernel_real = 16'shDB1C;  // [3,3]=-0.28819
         10'd 28: kernel_real = 16'shCD6F;  // [3,4]=-0.39505
         10'd 29: kernel_real = 16'shEA1D;  // [3,5]=-0.17098
         10'd 30: kernel_real = 16'sh2353;  // [3,6]=+0.27597
         10'd 31: kernel_real = 16'sh43A8;  // [3,7]=+0.52857
         10'd 32: kernel_real = 16'shD0F4;  // [4,0]=-0.36754
         10'd 33: kernel_real = 16'sh2A21;  // [4,1]=+0.32912
         10'd 34: kernel_real = 16'shD160;  // [4,2]=-0.36426
         10'd 35: kernel_real = 16'sh24E4;  // [4,3]=+0.28819
         10'd 36: kernel_real = 16'shCD6F;  // [4,4]=-0.39505
         10'd 37: kernel_real = 16'sh15E3;  // [4,5]=+0.17098
         10'd 38: kernel_real = 16'sh2353;  // [4,6]=+0.27597
         10'd 39: kernel_real = 16'shBC58;  // [4,7]=-0.52857
         10'd 40: kernel_real = 16'shD0C9;  // [5,0]=-0.36887
         10'd 41: kernel_real = 16'sh27FB;  // [5,1]=+0.31233
         10'd 42: kernel_real = 16'sh2C5A;  // [5,2]=+0.34648
         10'd 43: kernel_real = 16'sh3B2E;  // [5,3]=+0.46234
         10'd 44: kernel_real = 16'shDDDE;  // [5,4]=-0.26667
         10'd 45: kernel_real = 16'shE444;  // [5,5]=-0.21669
         10'd 46: kernel_real = 16'shCACC;  // [5,6]=-0.41566
         10'd 47: kernel_real = 16'sh3030;  // [5,7]=+0.37647
         10'd 48: kernel_real = 16'shD2A2;  // [6,0]=-0.35442
         10'd 49: kernel_real = 16'shCFE2;  // [6,1]=-0.37591
         10'd 50: kernel_real = 16'sh2BF1;  // [6,2]=+0.34328
         10'd 51: kernel_real = 16'sh1E9B;  // [6,3]=+0.23911
         10'd 52: kernel_real = 16'sh2642;  // [6,4]=+0.29888
         10'd 53: kernel_real = 16'shC21A;  // [6,5]=-0.48358
         10'd 54: kernel_real = 16'sh3457;  // [6,6]=+0.40892
         10'd 55: kernel_real = 16'shDEB4;  // [6,7]=-0.26013
         10'd 56: kernel_real = 16'shD6E0;  // [7,0]=-0.32130
         10'd 57: kernel_real = 16'shCDF7;  // [7,1]=-0.39091
         10'd 58: kernel_real = 16'shD1F4;  // [7,2]=-0.35975
         10'd 59: kernel_real = 16'sh30E9;  // [7,3]=+0.38211
         10'd 60: kernel_real = 16'sh36D5;  // [7,4]=+0.42836
         10'd 61: kernel_real = 16'sh37CA;  // [7,5]=+0.43585
         10'd 62: kernel_real = 16'shDAEF;  // [7,6]=-0.28957
         10'd 63: kernel_real = 16'sh0D8F;  // [7,7]=+0.10592
         // MODE 1: RFT-Fibonacci (unitarity: 1.09e-13)
         10'd 64: kernel_real = 16'sh0F14;  // [0,0]=+0.11779
         10'd 65: kernel_real = 16'shDA6C;  // [0,1]=-0.29359
         10'd 66: kernel_real = 16'sh3CCA;  // [0,2]=+0.47490
         10'd 67: kernel_real = 16'shCBCE;  // [0,3]=-0.40779
         10'd 68: kernel_real = 16'sh2ECE;  // [0,4]=+0.36565
         10'd 69: kernel_real = 16'shD2DC;  // [0,5]=-0.35265
         10'd 70: kernel_real = 16'sh2D99;  // [0,6]=+0.35622
         10'd 71: kernel_real = 16'sh2CEB;  // [0,7]=+0.35093
         10'd 72: kernel_real = 16'sh20AB;  // [1,0]=+0.25521
         10'd 73: kernel_real = 16'shCBD7;  // [1,1]=-0.40749
         10'd 74: kernel_real = 16'sh39DB;  // [1,2]=+0.45198
         10'd 75: kernel_real = 16'sh255C;  // [1,3]=+0.29187
         10'd 76: kernel_real = 16'shD689;  // [1,4]=-0.32394
         10'd 77: kernel_real = 16'shD314;  // [1,5]=-0.35094
         10'd 78: kernel_real = 16'shD2A1;  // [1,6]=-0.35446
         10'd 79: kernel_real = 16'shD2A3;  // [1,7]=-0.35441
         10'd 80: kernel_real = 16'sh3167;  // [2,0]=+0.38595
         10'd 81: kernel_real = 16'shCC19;  // [2,1]=-0.40547
         10'd 82: kernel_real = 16'shE9A5;  // [2,2]=-0.17465
         10'd 83: kernel_real = 16'sh252D;  // [2,3]=+0.29043
         10'd 84: kernel_real = 16'sh38E0;  // [2,4]=+0.44433
         10'd 85: kernel_real = 16'sh2D5E;  // [2,5]=+0.35443
         10'd 86: kernel_real = 16'shD316;  // [2,6]=-0.35088
         10'd 87: kernel_real = 16'sh2D5E;  // [2,7]=+0.35443
         10'd 88: kernel_real = 16'sh42C3;  // [3,0]=+0.52157
         10'd 89: kernel_real = 16'shDB0C;  // [3,1]=-0.28870
         10'd 90: kernel_real = 16'shE681;  // [3,2]=-0.19918
         10'd 91: kernel_real = 16'shCC23;  // [3,3]=-0.40518
         10'd 92: kernel_real = 16'shDFA2;  // [3,4]=-0.25286
         10'd 93: kernel_real = 16'sh2D97;  // [3,5]=+0.35617
         10'd 94: kernel_real = 16'sh2D23;  // [3,6]=+0.35264
         10'd 95: kernel_real = 16'shD2A2;  // [3,7]=-0.35444
         10'd 96: kernel_real = 16'sh42C3;  // [4,0]=+0.52157
         10'd 97: kernel_real = 16'sh24F4;  // [4,1]=+0.28870
         10'd 98: kernel_real = 16'shE681;  // [4,2]=-0.19918
         10'd 99: kernel_real = 16'sh33DD;  // [4,3]=+0.40518
         10'd100: kernel_real = 16'shDFA2;  // [4,4]=-0.25286
         10'd101: kernel_real = 16'shD269;  // [4,5]=-0.35617
         10'd102: kernel_real = 16'sh2D23;  // [4,6]=+0.35264
         10'd103: kernel_real = 16'sh2D5E;  // [4,7]=+0.35444
         10'd104: kernel_real = 16'sh3167;  // [5,0]=+0.38595
         10'd105: kernel_real = 16'sh33E7;  // [5,1]=+0.40547
         10'd106: kernel_real = 16'shE9A5;  // [5,2]=-0.17465
         10'd107: kernel_real = 16'shDAD3;  // [5,3]=-0.29043
         10'd108: kernel_real = 16'sh38E0;  // [5,4]=+0.44433
         10'd109: kernel_real = 16'shD2A2;  // [5,5]=-0.35443
         10'd110: kernel_real = 16'shD316;  // [5,6]=-0.35088
         10'd111: kernel_real = 16'shD2A2;  // [5,7]=-0.35443
         10'd112: kernel_real = 16'sh20AB;  // [6,0]=+0.25521
         10'd113: kernel_real = 16'sh3429;  // [6,1]=+0.40749
         10'd114: kernel_real = 16'sh39DB;  // [6,2]=+0.45198
         10'd115: kernel_real = 16'shDAA4;  // [6,3]=-0.29187
         10'd116: kernel_real = 16'shD689;  // [6,4]=-0.32394
         10'd117: kernel_real = 16'sh2CEC;  // [6,5]=+0.35094
         10'd118: kernel_real = 16'shD2A1;  // [6,6]=-0.35446
         10'd119: kernel_real = 16'sh2D5D;  // [6,7]=+0.35441
         10'd120: kernel_real = 16'sh0F14;  // [7,0]=+0.11779
         10'd121: kernel_real = 16'sh2594;  // [7,1]=+0.29359
         10'd122: kernel_real = 16'sh3CCA;  // [7,2]=+0.47490
         10'd123: kernel_real = 16'sh3432;  // [7,3]=+0.40779
         10'd124: kernel_real = 16'sh2ECE;  // [7,4]=+0.36565
         10'd125: kernel_real = 16'sh2D24;  // [7,5]=+0.35265
         10'd126: kernel_real = 16'sh2D99;  // [7,6]=+0.35622
         10'd127: kernel_real = 16'shD315;  // [7,7]=-0.35093
         // MODE 2: RFT-Harmonic (unitarity: 1.96e-15)
         10'd128: kernel_real = 16'sh2CF2;  // [0,0]=+0.35112
         10'd129: kernel_real = 16'shD2CA;  // [0,1]=-0.35321
         10'd130: kernel_real = 16'sh38CC;  // [0,2]=+0.44372
         10'd131: kernel_real = 16'sh4065;  // [0,3]=+0.50308
         10'd132: kernel_real = 16'sh228A;  // [0,4]=+0.26984
         10'd133: kernel_real = 16'shCAE7;  // [0,5]=-0.41482
         10'd134: kernel_real = 16'sh0A25;  // [0,6]=+0.07925
         10'd135: kernel_real = 16'sh1CDA;  // [0,7]=+0.22541
         10'd136: kernel_real = 16'shD2E2;  // [1,0]=-0.35247
         10'd137: kernel_real = 16'shD327;  // [1,1]=-0.35039
         10'd138: kernel_real = 16'shC2D3;  // [1,2]=-0.47794
         10'd139: kernel_real = 16'sh3485;  // [1,3]=+0.41030
         10'd140: kernel_real = 16'shC8E6;  // [1,4]=-0.43048
         10'd141: kernel_real = 16'shDDEC;  // [1,5]=-0.26622
         10'd142: kernel_real = 16'shDC44;  // [1,6]=-0.27916
         10'd143: kernel_real = 16'sh1307;  // [1,7]=+0.14866
         10'd144: kernel_real = 16'sh2D73;  // [2,0]=+0.35507
         10'd145: kernel_real = 16'shD276;  // [2,1]=-0.35577
         10'd146: kernel_real = 16'sh1A90;  // [2,2]=+0.20753
         10'd147: kernel_real = 16'sh12F5;  // [2,3]=+0.14809
         10'd148: kernel_real = 16'shD6D2;  // [2,4]=-0.32172
         10'd149: kernel_real = 16'sh3AB9;  // [2,5]=+0.45877
         10'd150: kernel_real = 16'shD3AE;  // [2,6]=-0.34625
         10'd151: kernel_real = 16'shC032;  // [2,7]=-0.49849
         10'd152: kernel_real = 16'shD27E;  // [3,0]=-0.35553
         10'd153: kernel_real = 16'shD295;  // [3,1]=-0.35482
         10'd154: kernel_real = 16'shE93D;  // [3,2]=-0.17782
         10'd155: kernel_real = 16'sh1E76;  // [3,3]=+0.23797
         10'd156: kernel_real = 16'sh2F9D;  // [3,4]=+0.37197
         10'd157: kernel_real = 16'sh1BA0;  // [3,5]=+0.21583
         10'd158: kernel_real = 16'sh45A1;  // [3,6]=+0.54397
         10'd159: kernel_real = 16'shC9E8;  // [3,7]=-0.42262
         10'd160: kernel_real = 16'sh2D82;  // [4,0]=+0.35553
         10'd161: kernel_real = 16'shD295;  // [4,1]=-0.35482
         10'd162: kernel_real = 16'shE93D;  // [4,2]=-0.17782
         10'd163: kernel_real = 16'shE18A;  // [4,3]=-0.23797
         10'd164: kernel_real = 16'shD063;  // [4,4]=-0.37197
         10'd165: kernel_real = 16'sh1BA0;  // [4,5]=+0.21583
         10'd166: kernel_real = 16'sh45A1;  // [4,6]=+0.54397
         10'd167: kernel_real = 16'sh3618;  // [4,7]=+0.42262
         10'd168: kernel_real = 16'shD28D;  // [5,0]=-0.35507
         10'd169: kernel_real = 16'shD276;  // [5,1]=-0.35577
         10'd170: kernel_real = 16'sh1A90;  // [5,2]=+0.20753
         10'd171: kernel_real = 16'shED0B;  // [5,3]=-0.14809
         10'd172: kernel_real = 16'sh292E;  // [5,4]=+0.32172
         10'd173: kernel_real = 16'sh3AB9;  // [5,5]=+0.45877
         10'd174: kernel_real = 16'shD3AE;  // [5,6]=-0.34625
         10'd175: kernel_real = 16'sh3FCE;  // [5,7]=+0.49849
         10'd176: kernel_real = 16'sh2D1E;  // [6,0]=+0.35247
         10'd177: kernel_real = 16'shD327;  // [6,1]=-0.35039
         10'd178: kernel_real = 16'shC2D3;  // [6,2]=-0.47794
         10'd179: kernel_real = 16'shCB7B;  // [6,3]=-0.41030
         10'd180: kernel_real = 16'sh371A;  // [6,4]=+0.43048
         10'd181: kernel_real = 16'shDDEC;  // [6,5]=-0.26622
         10'd182: kernel_real = 16'shDC44;  // [6,6]=-0.27916
         10'd183: kernel_real = 16'shECF9;  // [6,7]=-0.14866
         10'd184: kernel_real = 16'shD30E;  // [7,0]=-0.35112
         10'd185: kernel_real = 16'shD2CA;  // [7,1]=-0.35321
         10'd186: kernel_real = 16'sh38CC;  // [7,2]=+0.44372
         10'd187: kernel_real = 16'shBF9B;  // [7,3]=-0.50308
         10'd188: kernel_real = 16'shDD76;  // [7,4]=-0.26984
         10'd189: kernel_real = 16'shCAE7;  // [7,5]=-0.41482
         10'd190: kernel_real = 16'sh0A25;  // [7,6]=+0.07925
         10'd191: kernel_real = 16'shE326;  // [7,7]=-0.22541
         // MODE 3: RFT-Geometric (unitarity: 3.58e-15)
         10'd192: kernel_real = 16'shCA83;  // [0,0]=-0.41787
         10'd193: kernel_real = 16'sh0297;  // [0,1]=+0.02024
         10'd194: kernel_real = 16'sh501A;  // [0,2]=+0.62579
         10'd195: kernel_real = 16'shDBEC;  // [0,3]=-0.28187
         10'd196: kernel_real = 16'shDA12;  // [0,4]=-0.29632
         10'd197: kernel_real = 16'shF2CD;  // [0,5]=-0.10312
         10'd198: kernel_real = 16'sh3E17;  // [0,6]=+0.48508
         10'd199: kernel_real = 16'shEDD3;  // [0,7]=-0.14201
         10'd200: kernel_real = 16'shD827;  // [1,0]=-0.31131
         10'd201: kernel_real = 16'sh34C0;  // [1,1]=+0.41210
         10'd202: kernel_real = 16'sh2089;  // [1,2]=+0.25418
         10'd203: kernel_real = 16'sh1192;  // [1,3]=+0.13728
         10'd204: kernel_real = 16'sh3684;  // [1,4]=+0.42592
         10'd205: kernel_real = 16'sh4F01;  // [1,5]=+0.61723
         10'd206: kernel_real = 16'shF8AE;  // [1,6]=-0.05719
         10'd207: kernel_real = 16'sh2522;  // [1,7]=+0.29010
         10'd208: kernel_real = 16'sh1173;  // [2,0]=+0.13634
         10'd209: kernel_real = 16'sh4364;  // [2,1]=+0.52650
         10'd210: kernel_real = 16'shEA46;  // [2,2]=-0.16975
         10'd211: kernel_real = 16'sh459A;  // [2,3]=+0.54376
         10'd212: kernel_real = 16'shF0B9;  // [2,4]=-0.11935
         10'd213: kernel_real = 16'shFE6E;  // [2,5]=-0.01225
         10'd214: kernel_real = 16'sh3725;  // [2,6]=+0.43080
         10'd215: kernel_real = 16'shC9BC;  // [2,7]=-0.42396
         10'd216: kernel_real = 16'sh3AA4;  // [3,0]=+0.45813
         10'd217: kernel_real = 16'sh1D59;  // [3,1]=+0.22927
         10'd218: kernel_real = 16'shF058;  // [3,2]=-0.12231
         10'd219: kernel_real = 16'shD652;  // [3,3]=-0.32563
         10'd220: kernel_real = 16'shC470;  // [3,4]=-0.46534
         10'd221: kernel_real = 16'sh2A1D;  // [3,5]=+0.32901
         10'd222: kernel_real = 16'sh2340;  // [3,6]=+0.27538
         10'd223: kernel_real = 16'sh3B7B;  // [3,7]=+0.46469
         10'd224: kernel_real = 16'sh3AA4;  // [4,0]=+0.45813
         10'd225: kernel_real = 16'shE2A7;  // [4,1]=-0.22927
         10'd226: kernel_real = 16'sh0FA8;  // [4,2]=+0.12231
         10'd227: kernel_real = 16'shD652;  // [4,3]=-0.32563
         10'd228: kernel_real = 16'sh3B90;  // [4,4]=+0.46534
         10'd229: kernel_real = 16'sh2A1D;  // [4,5]=+0.32901
         10'd230: kernel_real = 16'sh2340;  // [4,6]=+0.27538
         10'd231: kernel_real = 16'shC485;  // [4,7]=-0.46469
         10'd232: kernel_real = 16'sh1173;  // [5,0]=+0.13634
         10'd233: kernel_real = 16'shBC9C;  // [5,1]=-0.52650
         10'd234: kernel_real = 16'sh15BA;  // [5,2]=+0.16975
         10'd235: kernel_real = 16'sh459A;  // [5,3]=+0.54376
         10'd236: kernel_real = 16'sh0F47;  // [5,4]=+0.11935
         10'd237: kernel_real = 16'shFE6E;  // [5,5]=-0.01225
         10'd238: kernel_real = 16'sh3725;  // [5,6]=+0.43080
         10'd239: kernel_real = 16'sh3644;  // [5,7]=+0.42396
         10'd240: kernel_real = 16'shD827;  // [6,0]=-0.31131
         10'd241: kernel_real = 16'shCB40;  // [6,1]=-0.41210
         10'd242: kernel_real = 16'shDF77;  // [6,2]=-0.25418
         10'd243: kernel_real = 16'sh1192;  // [6,3]=+0.13728
         10'd244: kernel_real = 16'shC97C;  // [6,4]=-0.42592
         10'd245: kernel_real = 16'sh4F01;  // [6,5]=+0.61723
         10'd246: kernel_real = 16'shF8AE;  // [6,6]=-0.05719
         10'd247: kernel_real = 16'shDADE;  // [6,7]=-0.29010
         10'd248: kernel_real = 16'shCA83;  // [7,0]=-0.41787
         10'd249: kernel_real = 16'shFD69;  // [7,1]=-0.02024
         10'd250: kernel_real = 16'shAFE6;  // [7,2]=-0.62579
         10'd251: kernel_real = 16'shDBEC;  // [7,3]=-0.28187
         10'd252: kernel_real = 16'sh25EE;  // [7,4]=+0.29632
         10'd253: kernel_real = 16'shF2CD;  // [7,5]=-0.10312
         10'd254: kernel_real = 16'sh3E17;  // [7,6]=+0.48508
         10'd255: kernel_real = 16'sh122D;  // [7,7]=+0.14201
         // MODE 4: RFT-Beating (unitarity: 3.09e-15)
         10'd256: kernel_real = 16'shD4AA;  // [0,0]=-0.33856
         10'd257: kernel_real = 16'sh3C3A;  // [0,1]=+0.47051
         10'd258: kernel_real = 16'sh1705;  // [0,2]=+0.17983
         10'd259: kernel_real = 16'shCD4B;  // [0,3]=-0.39615
         10'd260: kernel_real = 16'sh481A;  // [0,4]=+0.56329
         10'd261: kernel_real = 16'shE778;  // [0,5]=-0.19165
         10'd262: kernel_real = 16'sh1833;  // [0,6]=+0.18905
         10'd263: kernel_real = 16'shDAB1;  // [0,7]=-0.29147
         10'd264: kernel_real = 16'shD044;  // [1,0]=-0.37293
         10'd265: kernel_real = 16'shE536;  // [1,1]=-0.20930
         10'd266: kernel_real = 16'shC8A4;  // [1,2]=-0.43250
         10'd267: kernel_real = 16'shBFFE;  // [1,3]=-0.50007
         10'd268: kernel_real = 16'shE572;  // [1,4]=-0.20747
         10'd269: kernel_real = 16'sh39D4;  // [1,5]=+0.45179
         10'd270: kernel_real = 16'sh2E4C;  // [1,6]=+0.36170
         10'd271: kernel_real = 16'sh05BA;  // [1,7]=+0.04474
         10'd272: kernel_real = 16'shD594;  // [2,0]=-0.33142
         10'd273: kernel_real = 16'shEAAB;  // [2,1]=-0.16667
         10'd274: kernel_real = 16'sh3FF0;  // [2,2]=+0.49951
         10'd275: kernel_real = 16'shDA0A;  // [2,3]=-0.29657
         10'd276: kernel_real = 16'shD04B;  // [2,4]=-0.37270
         10'd277: kernel_real = 16'shC6AB;  // [2,5]=-0.44791
         10'd278: kernel_real = 16'sh055A;  // [2,6]=+0.04180
         10'd279: kernel_real = 16'sh36DA;  // [2,7]=+0.42854
         10'd280: kernel_real = 16'shD0B7;  // [3,0]=-0.36940
         10'd281: kernel_real = 16'sh3A3E;  // [3,1]=+0.45501
         10'd282: kernel_real = 16'shE96E;  // [3,2]=-0.17634
         10'd283: kernel_real = 16'sh0916;  // [3,3]=+0.07099
         10'd284: kernel_real = 16'sh0387;  // [3,4]=+0.02756
         10'd285: kernel_real = 16'sh1EF7;  // [3,5]=+0.24193
         10'd286: kernel_real = 16'shB648;  // [3,6]=-0.57592
         10'd287: kernel_real = 16'sh3D4E;  // [3,7]=+0.47895
         10'd288: kernel_real = 16'shD0B7;  // [4,0]=-0.36940
         10'd289: kernel_real = 16'shC5C2;  // [4,1]=-0.45501
         10'd290: kernel_real = 16'shE96E;  // [4,2]=-0.17634
         10'd291: kernel_real = 16'shF6EA;  // [4,3]=-0.07099
         10'd292: kernel_real = 16'sh0387;  // [4,4]=+0.02756
         10'd293: kernel_real = 16'shE109;  // [4,5]=-0.24193
         10'd294: kernel_real = 16'shB648;  // [4,6]=-0.57592
         10'd295: kernel_real = 16'shC2B2;  // [4,7]=-0.47895
         10'd296: kernel_real = 16'shD594;  // [5,0]=-0.33142
         10'd297: kernel_real = 16'sh1555;  // [5,1]=+0.16667
         10'd298: kernel_real = 16'sh3FF0;  // [5,2]=+0.49951
         10'd299: kernel_real = 16'sh25F6;  // [5,3]=+0.29657
         10'd300: kernel_real = 16'shD04B;  // [5,4]=-0.37270
         10'd301: kernel_real = 16'sh3955;  // [5,5]=+0.44791
         10'd302: kernel_real = 16'sh055A;  // [5,6]=+0.04180
         10'd303: kernel_real = 16'shC926;  // [5,7]=-0.42854
         10'd304: kernel_real = 16'shD044;  // [6,0]=-0.37293
         10'd305: kernel_real = 16'sh1ACA;  // [6,1]=+0.20930
         10'd306: kernel_real = 16'shC8A4;  // [6,2]=-0.43250
         10'd307: kernel_real = 16'sh4002;  // [6,3]=+0.50007
         10'd308: kernel_real = 16'shE572;  // [6,4]=-0.20747
         10'd309: kernel_real = 16'shC62C;  // [6,5]=-0.45179
         10'd310: kernel_real = 16'sh2E4C;  // [6,6]=+0.36170
         10'd311: kernel_real = 16'shFA46;  // [6,7]=-0.04474
         10'd312: kernel_real = 16'shD4AA;  // [7,0]=-0.33856
         10'd313: kernel_real = 16'shC3C6;  // [7,1]=-0.47051
         10'd314: kernel_real = 16'sh1705;  // [7,2]=+0.17983
         10'd315: kernel_real = 16'sh32B5;  // [7,3]=+0.39615
         10'd316: kernel_real = 16'sh481A;  // [7,4]=+0.56329
         10'd317: kernel_real = 16'sh1888;  // [7,5]=+0.19165
         10'd318: kernel_real = 16'sh1833;  // [7,6]=+0.18905
         10'd319: kernel_real = 16'sh254F;  // [7,7]=+0.29147
         // MODE 5: RFT-Phyllotaxis (unitarity: 4.38e-15)
         10'd320: kernel_real = 16'sh257D;  // [0,0]=+0.29289
         10'd321: kernel_real = 16'sh3C92;  // [0,1]=+0.47321
         10'd322: kernel_real = 16'shDA77;  // [0,2]=-0.29323
         10'd323: kernel_real = 16'sh15C6;  // [0,3]=+0.17012
         10'd324: kernel_real = 16'sh3601;  // [0,4]=+0.42189
         10'd325: kernel_real = 16'shF1EC;  // [0,5]=-0.10999
         10'd326: kernel_real = 16'sh3F5F;  // [0,6]=+0.49508
         10'd327: kernel_real = 16'shD014;  // [0,7]=-0.37440
         10'd328: kernel_real = 16'shC0D5;  // [1,0]=-0.49350
         10'd329: kernel_real = 16'shF8D4;  // [1,1]=-0.05603
         10'd330: kernel_real = 16'sh00F8;  // [1,2]=+0.00757
         10'd331: kernel_real = 16'sh2FFB;  // [1,3]=+0.37484
         10'd332: kernel_real = 16'sh1ED3;  // [1,4]=+0.24080
         10'd333: kernel_real = 16'sh54CB;  // [1,5]=+0.66243
         10'd334: kernel_real = 16'shF4FE;  // [1,6]=-0.08601
         10'd335: kernel_real = 16'shD5D4;  // [1,7]=-0.32947
         10'd336: kernel_real = 16'sh318E;  // [2,0]=+0.38714
         10'd337: kernel_real = 16'shD4F4;  // [2,1]=-0.33629
         10'd338: kernel_real = 16'sh236B;  // [2,2]=+0.27669
         10'd339: kernel_real = 16'sh121F;  // [2,3]=+0.14157
         10'd340: kernel_real = 16'sh419F;  // [2,4]=+0.51268
         10'd341: kernel_real = 16'shE41A;  // [2,5]=-0.21797
         10'd342: kernel_real = 16'shC076;  // [2,6]=-0.49641
         10'd343: kernel_real = 16'shDAFA;  // [2,7]=-0.28923
         10'd344: kernel_real = 16'shED8B;  // [3,0]=-0.14420
         10'd345: kernel_real = 16'sh332D;  // [3,1]=+0.39980
         10'd346: kernel_real = 16'sh4A5A;  // [3,2]=+0.58086
         10'd347: kernel_real = 16'shB8AC;  // [3,3]=-0.55724
         10'd348: kernel_real = 16'shFB9A;  // [3,4]=-0.03437
         10'd349: kernel_real = 16'sh0514;  // [3,5]=+0.03967
         10'd350: kernel_real = 16'shFBCC;  // [3,6]=-0.03283
         10'd351: kernel_real = 16'shCB98;  // [3,7]=-0.40941
         10'd352: kernel_real = 16'shED8B;  // [4,0]=-0.14420
         10'd353: kernel_real = 16'shCCD3;  // [4,1]=-0.39980
         10'd354: kernel_real = 16'shB5A6;  // [4,2]=-0.58086
         10'd355: kernel_real = 16'shB8AC;  // [4,3]=-0.55724
         10'd356: kernel_real = 16'sh0466;  // [4,4]=+0.03437
         10'd357: kernel_real = 16'shFAEC;  // [4,5]=-0.03967
         10'd358: kernel_real = 16'shFBCC;  // [4,6]=-0.03283
         10'd359: kernel_real = 16'shCB98;  // [4,7]=-0.40941
         10'd360: kernel_real = 16'sh318E;  // [5,0]=+0.38714
         10'd361: kernel_real = 16'sh2B0C;  // [5,1]=+0.33629
         10'd362: kernel_real = 16'shDC95;  // [5,2]=-0.27669
         10'd363: kernel_real = 16'sh121F;  // [5,3]=+0.14157
         10'd364: kernel_real = 16'shBE61;  // [5,4]=-0.51268
         10'd365: kernel_real = 16'sh1BE6;  // [5,5]=+0.21797
         10'd366: kernel_real = 16'shC076;  // [5,6]=-0.49641
         10'd367: kernel_real = 16'shDAFA;  // [5,7]=-0.28923
         10'd368: kernel_real = 16'shC0D5;  // [6,0]=-0.49350
         10'd369: kernel_real = 16'sh072C;  // [6,1]=+0.05603
         10'd370: kernel_real = 16'shFF08;  // [6,2]=-0.00757
         10'd371: kernel_real = 16'sh2FFB;  // [6,3]=+0.37484
         10'd372: kernel_real = 16'shE12D;  // [6,4]=-0.24080
         10'd373: kernel_real = 16'shAB35;  // [6,5]=-0.66243
         10'd374: kernel_real = 16'shF4FE;  // [6,6]=-0.08601
         10'd375: kernel_real = 16'shD5D4;  // [6,7]=-0.32947
         10'd376: kernel_real = 16'sh257D;  // [7,0]=+0.29289
         10'd377: kernel_real = 16'shC36E;  // [7,1]=-0.47321
         10'd378: kernel_real = 16'sh2589;  // [7,2]=+0.29323
         10'd379: kernel_real = 16'sh15C6;  // [7,3]=+0.17012
         10'd380: kernel_real = 16'shC9FF;  // [7,4]=-0.42189
         10'd381: kernel_real = 16'sh0E14;  // [7,5]=+0.10999
         10'd382: kernel_real = 16'sh3F5F;  // [7,6]=+0.49508
         10'd383: kernel_real = 16'shD014;  // [7,7]=-0.37440
         // MODE 6: RFT-Cascade (unitarity: 1.51e-15)
         10'd384: kernel_real = 16'shDA01;  // [0,0]=-0.29686
         10'd385: kernel_real = 16'sh3E27;  // [0,1]=+0.48556
         10'd386: kernel_real = 16'sh3755;  // [0,2]=+0.43227
         10'd387: kernel_real = 16'shF620;  // [0,3]=-0.07714
         10'd388: kernel_real = 16'sh3C33;  // [0,4]=+0.47030
         10'd389: kernel_real = 16'shEF29;  // [0,5]=-0.13157
         10'd390: kernel_real = 16'sh07EE;  // [0,6]=+0.06194
         10'd391: kernel_real = 16'sh3ED6;  // [0,7]=+0.49089
         10'd392: kernel_real = 16'shD578;  // [1,0]=-0.33228
         10'd393: kernel_real = 16'sh3E8D;  // [1,1]=+0.48867
         10'd394: kernel_real = 16'shCE99;  // [1,2]=-0.38596
         10'd395: kernel_real = 16'shF39D;  // [1,3]=-0.09677
         10'd396: kernel_real = 16'sh0A68;  // [1,4]=+0.08131
         10'd397: kernel_real = 16'sh216A;  // [1,5]=+0.26103
         10'd398: kernel_real = 16'sh3DEC;  // [1,6]=+0.48375
         10'd399: kernel_real = 16'shC924;  // [1,7]=-0.42860
         10'd400: kernel_real = 16'shCEFB;  // [2,0]=-0.38297
         10'd401: kernel_real = 16'shF9A2;  // [2,1]=-0.04973
         10'd402: kernel_real = 16'shDAD3;  // [2,2]=-0.29045
         10'd403: kernel_real = 16'shB416;  // [2,3]=-0.59309
         10'd404: kernel_real = 16'sh0BD5;  // [2,4]=+0.09244
         10'd405: kernel_real = 16'shD280;  // [2,5]=-0.35548
         10'd406: kernel_real = 16'shBEAE;  // [2,6]=-0.51032
         10'd407: kernel_real = 16'shEE2C;  // [2,7]=-0.13929
         10'd408: kernel_real = 16'shCDA3;  // [3,0]=-0.39345
         10'd409: kernel_real = 16'shEC9A;  // [3,1]=-0.15154
         10'd410: kernel_real = 16'sh2429;  // [3,2]=+0.28251
         10'd411: kernel_real = 16'shD155;  // [3,3]=-0.36459
         10'd412: kernel_real = 16'shBE46;  // [3,4]=-0.51348
         10'd413: kernel_real = 16'sh44B7;  // [3,5]=+0.53683
         10'd414: kernel_real = 16'sh054E;  // [3,6]=+0.04146
         10'd415: kernel_real = 16'sh1E45;  // [3,7]=+0.23649
         10'd416: kernel_real = 16'shCDA3;  // [4,0]=-0.39345
         10'd417: kernel_real = 16'sh1366;  // [4,1]=+0.15154
         10'd418: kernel_real = 16'sh2429;  // [4,2]=+0.28251
         10'd419: kernel_real = 16'sh2EAB;  // [4,3]=+0.36459
         10'd420: kernel_real = 16'shBE46;  // [4,4]=-0.51348
         10'd421: kernel_real = 16'shBB49;  // [4,5]=-0.53683
         10'd422: kernel_real = 16'sh054E;  // [4,6]=+0.04146
         10'd423: kernel_real = 16'shE1BB;  // [4,7]=-0.23649
         10'd424: kernel_real = 16'shCEFB;  // [5,0]=-0.38297
         10'd425: kernel_real = 16'sh065E;  // [5,1]=+0.04973
         10'd426: kernel_real = 16'shDAD3;  // [5,2]=-0.29045
         10'd427: kernel_real = 16'sh4BEA;  // [5,3]=+0.59309
         10'd428: kernel_real = 16'sh0BD5;  // [5,4]=+0.09244
         10'd429: kernel_real = 16'sh2D80;  // [5,5]=+0.35548
         10'd430: kernel_real = 16'shBEAE;  // [5,6]=-0.51032
         10'd431: kernel_real = 16'sh11D4;  // [5,7]=+0.13929
         10'd432: kernel_real = 16'shD578;  // [6,0]=-0.33228
         10'd433: kernel_real = 16'shC173;  // [6,1]=-0.48867
         10'd434: kernel_real = 16'shCE99;  // [6,2]=-0.38596
         10'd435: kernel_real = 16'sh0C63;  // [6,3]=+0.09677
         10'd436: kernel_real = 16'sh0A68;  // [6,4]=+0.08131
         10'd437: kernel_real = 16'shDE96;  // [6,5]=-0.26103
         10'd438: kernel_real = 16'sh3DEC;  // [6,6]=+0.48375
         10'd439: kernel_real = 16'sh36DC;  // [6,7]=+0.42860
         10'd440: kernel_real = 16'shDA01;  // [7,0]=-0.29686
         10'd441: kernel_real = 16'shC1D9;  // [7,1]=-0.48556
         10'd442: kernel_real = 16'sh3755;  // [7,2]=+0.43227
         10'd443: kernel_real = 16'sh09E0;  // [7,3]=+0.07714
         10'd444: kernel_real = 16'sh3C33;  // [7,4]=+0.47030
         10'd445: kernel_real = 16'sh10D7;  // [7,5]=+0.13157
         10'd446: kernel_real = 16'sh07EE;  // [7,6]=+0.06194
         10'd447: kernel_real = 16'shC12A;  // [7,7]=-0.49089
         // MODE 7: RFT-Hybrid-DCT (unitarity: 1.12e-15)
         10'd448: kernel_real = 16'shD2BF;  // [0,0]=-0.35355
         10'd449: kernel_real = 16'shD2BF;  // [0,1]=-0.35355
         10'd450: kernel_real = 16'shD2BF;  // [0,2]=-0.35355
         10'd451: kernel_real = 16'shD2BF;  // [0,3]=-0.35355
         10'd452: kernel_real = 16'shD2BF;  // [0,4]=-0.35355
         10'd453: kernel_real = 16'shD2BF;  // [0,5]=-0.35355
         10'd454: kernel_real = 16'shD2BF;  // [0,6]=-0.35355
         10'd455: kernel_real = 16'shD2BF;  // [0,7]=-0.35355
         10'd456: kernel_real = 16'shC13B;  // [1,0]=-0.49039
         10'd457: kernel_real = 16'shCAC9;  // [1,1]=-0.41573
         10'd458: kernel_real = 16'shDC72;  // [1,2]=-0.27779
         10'd459: kernel_real = 16'shF384;  // [1,3]=-0.09755
         10'd460: kernel_real = 16'sh0C7C;  // [1,4]=+0.09755
         10'd461: kernel_real = 16'sh238E;  // [1,5]=+0.27779
         10'd462: kernel_real = 16'sh3537;  // [1,6]=+0.41573
         10'd463: kernel_real = 16'sh3EC5;  // [1,7]=+0.49039
         10'd464: kernel_real = 16'sh3B21;  // [2,0]=+0.46194
         10'd465: kernel_real = 16'sh187E;  // [2,1]=+0.19134
         10'd466: kernel_real = 16'shE782;  // [2,2]=-0.19134
         10'd467: kernel_real = 16'shC4DF;  // [2,3]=-0.46194
         10'd468: kernel_real = 16'shC4DF;  // [2,4]=-0.46194
         10'd469: kernel_real = 16'shE782;  // [2,5]=-0.19134
         10'd470: kernel_real = 16'sh187E;  // [2,6]=+0.19134
         10'd471: kernel_real = 16'sh3B21;  // [2,7]=+0.46194
         10'd472: kernel_real = 16'sh3537;  // [3,0]=+0.41573
         10'd473: kernel_real = 16'shF384;  // [3,1]=-0.09755
         10'd474: kernel_real = 16'shC13B;  // [3,2]=-0.49039
         10'd475: kernel_real = 16'shDC72;  // [3,3]=-0.27779
         10'd476: kernel_real = 16'sh238E;  // [3,4]=+0.27779
         10'd477: kernel_real = 16'sh3EC5;  // [3,5]=+0.49039
         10'd478: kernel_real = 16'sh0C7C;  // [3,6]=+0.09755
         10'd479: kernel_real = 16'shCAC9;  // [3,7]=-0.41573
         10'd480: kernel_real = 16'shDCA5;  // [4,0]=-0.27622
         10'd481: kernel_real = 16'sh3CC9;  // [4,1]=+0.47488
         10'd482: kernel_real = 16'shDB65;  // [4,2]=-0.28599
         10'd483: kernel_real = 16'sh2899;  // [4,3]=+0.31717
         10'd484: kernel_real = 16'shC270;  // [4,4]=-0.48095
         10'd485: kernel_real = 16'sh1377;  // [4,5]=+0.15206
         10'd486: kernel_real = 16'sh342D;  // [4,6]=+0.40762
         10'd487: kernel_real = 16'shD881;  // [4,7]=-0.30857
         10'd488: kernel_real = 16'sh08C9;  // [5,0]=+0.06862
         10'd489: kernel_real = 16'shE1B8;  // [5,1]=-0.23657
         10'd490: kernel_real = 16'sh2957;  // [5,2]=+0.32298
         10'd491: kernel_real = 16'shE7E2;  // [5,3]=-0.18841
         10'd492: kernel_real = 16'sh0E3F;  // [5,4]=+0.11130
         10'd493: kernel_real = 16'shCC32;  // [5,5]=-0.40473
         10'd494: kernel_real = 16'sh593F;  // [5,6]=+0.69722
         10'd495: kernel_real = 16'shD096;  // [5,7]=-0.37041
         10'd496: kernel_real = 16'shE40B;  // [6,0]=-0.21840
         10'd497: kernel_real = 16'sh0BC1;  // [6,1]=+0.09184
         10'd498: kernel_real = 16'sh4783;  // [6,2]=+0.55867
         10'd499: kernel_real = 16'shBE16;  // [6,3]=-0.51494
         10'd500: kernel_real = 16'shDE63;  // [6,4]=-0.26260
         10'd501: kernel_real = 16'sh439F;  // [6,5]=+0.52829
         10'd502: kernel_real = 16'shF828;  // [6,6]=-0.06129
         10'd503: kernel_real = 16'shF071;  // [6,7]=-0.12156
         10'd504: kernel_real = 16'shD3BC;  // [7,0]=-0.34581
         10'd505: kernel_real = 16'sh4D73;  // [7,1]=+0.60509
         10'd506: kernel_real = 16'shECCB;  // [7,2]=-0.15007
         10'd507: kernel_real = 16'shCAA9;  // [7,3]=-0.41673
         10'd508: kernel_real = 16'sh418D;  // [7,4]=+0.51212
         10'd509: kernel_real = 16'shE20F;  // [7,5]=-0.23392
         10'd510: kernel_real = 16'shFF86;  // [7,6]=-0.00373
         10'd511: kernel_real = 16'sh043B;  // [7,7]=+0.03306
         // MODE 8: RFT-Manifold (unitarity: 2.26e-15)
         10'd512: kernel_real = 16'shDB61;  // [0,0]=-0.28609
         10'd513: kernel_real = 16'shC56B;  // [0,1]=-0.45768
         10'd514: kernel_real = 16'shD756;  // [0,2]=-0.31769
         10'd515: kernel_real = 16'shC6F6;  // [0,3]=-0.44562
         10'd516: kernel_real = 16'shF79B;  // [0,4]=-0.06557
         10'd517: kernel_real = 16'shE740;  // [0,5]=-0.19336
         10'd518: kernel_real = 16'shE219;  // [0,6]=-0.23360
         10'd519: kernel_real = 16'sh479A;  // [0,7]=+0.55940
         10'd520: kernel_real = 16'shC3A4;  // [1,0]=-0.47155
         10'd521: kernel_real = 16'shF575;  // [1,1]=-0.08238
         10'd522: kernel_real = 16'sh38C2;  // [1,2]=+0.44343
         10'd523: kernel_real = 16'shEFC1;  // [1,3]=-0.12691
         10'd524: kernel_real = 16'sh2400;  // [1,4]=+0.28125
         10'd525: kernel_real = 16'shDC8C;  // [1,5]=-0.27698
         10'd526: kernel_real = 16'sh50FE;  // [1,6]=+0.63276
         10'd527: kernel_real = 16'sh0596;  // [1,7]=+0.04363
         10'd528: kernel_real = 16'shCA88;  // [2,0]=-0.41774
         10'd529: kernel_real = 16'sh2624;  // [2,1]=+0.29797
         10'd530: kernel_real = 16'shD070;  // [2,2]=-0.37159
         10'd531: kernel_real = 16'sh0870;  // [2,3]=+0.06592
         10'd532: kernel_real = 16'shF9E2;  // [2,4]=-0.04778
         10'd533: kernel_real = 16'shB2F8;  // [2,5]=-0.60180
         10'd534: kernel_real = 16'shE4F0;  // [2,6]=-0.21142
         10'd535: kernel_real = 16'shC8ED;  // [2,7]=-0.43027
         10'd536: kernel_real = 16'shED53;  // [3,0]=-0.14590
         10'd537: kernel_real = 16'sh3884;  // [3,1]=+0.44154
         10'd538: kernel_real = 16'sh2079;  // [3,2]=+0.25370
         10'd539: kernel_real = 16'shBC27;  // [3,3]=-0.53007
         10'd540: kernel_real = 16'shAD9C;  // [3,4]=-0.64367
         10'd541: kernel_real = 16'sh13B7;  // [3,5]=+0.15403
         10'd542: kernel_real = 16'sh0261;  // [3,6]=+0.01859
         10'd543: kernel_real = 16'shFF3C;  // [3,7]=-0.00598
         10'd544: kernel_real = 16'sh12AD;  // [4,0]=+0.14590
         10'd545: kernel_real = 16'sh3884;  // [4,1]=+0.44154
         10'd546: kernel_real = 16'shDF87;  // [4,2]=-0.25370
         10'd547: kernel_real = 16'shBC27;  // [4,3]=-0.53007
         10'd548: kernel_real = 16'sh5264;  // [4,4]=+0.64367
         10'd549: kernel_real = 16'sh13B7;  // [4,5]=+0.15403
         10'd550: kernel_real = 16'sh0261;  // [4,6]=+0.01859
         10'd551: kernel_real = 16'sh00C4;  // [4,7]=+0.00598
         10'd552: kernel_real = 16'sh3578;  // [5,0]=+0.41774
         10'd553: kernel_real = 16'sh2624;  // [5,1]=+0.29797
         10'd554: kernel_real = 16'sh2F90;  // [5,2]=+0.37159
         10'd555: kernel_real = 16'sh0870;  // [5,3]=+0.06592
         10'd556: kernel_real = 16'sh061E;  // [5,4]=+0.04778
         10'd557: kernel_real = 16'shB2F8;  // [5,5]=-0.60180
         10'd558: kernel_real = 16'shE4F0;  // [5,6]=-0.21142
         10'd559: kernel_real = 16'sh3713;  // [5,7]=+0.43027
         10'd560: kernel_real = 16'sh3C5C;  // [6,0]=+0.47155
         10'd561: kernel_real = 16'shF575;  // [6,1]=-0.08238
         10'd562: kernel_real = 16'shC73E;  // [6,2]=-0.44343
         10'd563: kernel_real = 16'shEFC1;  // [6,3]=-0.12691
         10'd564: kernel_real = 16'shDC00;  // [6,4]=-0.28125
         10'd565: kernel_real = 16'shDC8C;  // [6,5]=-0.27698
         10'd566: kernel_real = 16'sh50FE;  // [6,6]=+0.63276
         10'd567: kernel_real = 16'shFA6A;  // [6,7]=-0.04363
         10'd568: kernel_real = 16'sh249F;  // [7,0]=+0.28609
         10'd569: kernel_real = 16'shC56B;  // [7,1]=-0.45768
         10'd570: kernel_real = 16'sh28AA;  // [7,2]=+0.31769
         10'd571: kernel_real = 16'shC6F6;  // [7,3]=-0.44562
         10'd572: kernel_real = 16'sh0865;  // [7,4]=+0.06557
         10'd573: kernel_real = 16'shE740;  // [7,5]=-0.19336
         10'd574: kernel_real = 16'shE219;  // [7,6]=-0.23360
         10'd575: kernel_real = 16'shB866;  // [7,7]=-0.55940
         // MODE 9: RFT-Euler (unitarity: 3.02e-15)
         10'd576: kernel_real = 16'shC097;  // [0,0]=-0.49539
         10'd577: kernel_real = 16'sh1453;  // [0,1]=+0.15879
         10'd578: kernel_real = 16'shF140;  // [0,2]=-0.11524
         10'd579: kernel_real = 16'shD1DE;  // [0,3]=-0.36041
         10'd580: kernel_real = 16'sh4E0F;  // [0,4]=+0.60983
         10'd581: kernel_real = 16'shF02F;  // [0,5]=-0.12358
         10'd582: kernel_real = 16'sh2651;  // [0,6]=+0.29935
         10'd583: kernel_real = 16'shD5A9;  // [0,7]=-0.33079
         10'd584: kernel_real = 16'shF6DE;  // [1,0]=-0.07135
         10'd585: kernel_real = 16'sh471A;  // [1,1]=+0.55548
         10'd586: kernel_real = 16'shEA82;  // [1,2]=-0.16791
         10'd587: kernel_real = 16'shC8AA;  // [1,3]=-0.43231
         10'd588: kernel_real = 16'shD8D7;  // [1,4]=-0.30593
         10'd589: kernel_real = 16'sh26EC;  // [1,5]=+0.30407
         10'd590: kernel_real = 16'sh21C9;  // [1,6]=+0.26393
         10'd591: kernel_real = 16'sh3B6E;  // [1,7]=+0.46428
         10'd592: kernel_real = 16'sh3542;  // [2,0]=+0.41607
         10'd593: kernel_real = 16'sh1BD2;  // [2,1]=+0.21735
         10'd594: kernel_real = 16'shCF98;  // [2,2]=-0.37817
         10'd595: kernel_real = 16'shCD2D;  // [2,3]=-0.39705
         10'd596: kernel_real = 16'sh1181;  // [2,4]=+0.13676
         10'd597: kernel_real = 16'shCB90;  // [2,5]=-0.40967
         10'd598: kernel_real = 16'shBAF2;  // [2,6]=-0.53949
         10'd599: kernel_real = 16'shFB35;  // [2,7]=-0.03746
         10'd600: kernel_real = 16'sh2360;  // [3,0]=+0.27638
         10'd601: kernel_real = 16'shD3D9;  // [3,1]=-0.34495
         10'd602: kernel_real = 16'shB81A;  // [3,2]=-0.56171
         10'd603: kernel_real = 16'shEB89;  // [3,3]=-0.15988
         10'd604: kernel_real = 16'shEFE8;  // [3,4]=-0.12574
         10'd605: kernel_real = 16'sh3CA4;  // [3,5]=+0.47375
         10'd606: kernel_real = 16'sh1C88;  // [3,6]=+0.22290
         10'd607: kernel_real = 16'shCAAA;  // [3,7]=-0.41667
         10'd608: kernel_real = 16'shDCA0;  // [4,0]=-0.27638
         10'd609: kernel_real = 16'shD3D9;  // [4,1]=-0.34495
         10'd610: kernel_real = 16'shB81A;  // [4,2]=-0.56171
         10'd611: kernel_real = 16'sh1477;  // [4,3]=+0.15988
         10'd612: kernel_real = 16'shEFE8;  // [4,4]=-0.12574
         10'd613: kernel_real = 16'shC35C;  // [4,5]=-0.47375
         10'd614: kernel_real = 16'sh1C88;  // [4,6]=+0.22290
         10'd615: kernel_real = 16'sh3556;  // [4,7]=+0.41667
         10'd616: kernel_real = 16'shCABE;  // [5,0]=-0.41607
         10'd617: kernel_real = 16'sh1BD2;  // [5,1]=+0.21735
         10'd618: kernel_real = 16'shCF98;  // [5,2]=-0.37817
         10'd619: kernel_real = 16'sh32D3;  // [5,3]=+0.39705
         10'd620: kernel_real = 16'sh1181;  // [5,4]=+0.13676
         10'd621: kernel_real = 16'sh3470;  // [5,5]=+0.40967
         10'd622: kernel_real = 16'shBAF2;  // [5,6]=-0.53949
         10'd623: kernel_real = 16'sh04CB;  // [5,7]=+0.03746
         10'd624: kernel_real = 16'sh0922;  // [6,0]=+0.07135
         10'd625: kernel_real = 16'sh471A;  // [6,1]=+0.55548
         10'd626: kernel_real = 16'shEA82;  // [6,2]=-0.16791
         10'd627: kernel_real = 16'sh3756;  // [6,3]=+0.43231
         10'd628: kernel_real = 16'shD8D7;  // [6,4]=-0.30593
         10'd629: kernel_real = 16'shD914;  // [6,5]=-0.30407
         10'd630: kernel_real = 16'sh21C9;  // [6,6]=+0.26393
         10'd631: kernel_real = 16'shC492;  // [6,7]=-0.46428
         10'd632: kernel_real = 16'sh3F69;  // [7,0]=+0.49539
         10'd633: kernel_real = 16'sh1453;  // [7,1]=+0.15879
         10'd634: kernel_real = 16'shF140;  // [7,2]=-0.11524
         10'd635: kernel_real = 16'sh2E22;  // [7,3]=+0.36041
         10'd636: kernel_real = 16'sh4E0F;  // [7,4]=+0.60983
         10'd637: kernel_real = 16'sh0FD1;  // [7,5]=+0.12358
         10'd638: kernel_real = 16'sh2651;  // [7,6]=+0.29935
         10'd639: kernel_real = 16'sh2A57;  // [7,7]=+0.33079
         // MODE 10: RFT-PhaseCoh (unitarity: 3.36e-15)
         10'd640: kernel_real = 16'shD423;  // [0,0]=-0.34269
         10'd641: kernel_real = 16'shC33E;  // [0,1]=-0.47467
         10'd642: kernel_real = 16'sh15F1;  // [0,2]=+0.17143
         10'd643: kernel_real = 16'shCAD1;  // [0,3]=-0.41548
         10'd644: kernel_real = 16'shC653;  // [0,4]=-0.45060
         10'd645: kernel_real = 16'sh087B;  // [0,5]=+0.06626
         10'd646: kernel_real = 16'sh2801;  // [0,6]=+0.31252
         10'd647: kernel_real = 16'sh3199;  // [0,7]=+0.38747
         10'd648: kernel_real = 16'shD207;  // [1,0]=-0.35915
         10'd649: kernel_real = 16'sh19DA;  // [1,1]=+0.20196
         10'd650: kernel_real = 16'shC6F1;  // [1,2]=-0.44577
         10'd651: kernel_real = 16'shBE48;  // [1,3]=-0.51343
         10'd652: kernel_real = 16'sh29A3;  // [1,4]=+0.32528
         10'd653: kernel_real = 16'sh1504;  // [1,5]=+0.16420
         10'd654: kernel_real = 16'shCB70;  // [1,6]=-0.41066
         10'd655: kernel_real = 16'sh2102;  // [1,7]=+0.25786
         10'd656: kernel_real = 16'shD35C;  // [2,0]=-0.34876
         10'd657: kernel_real = 16'sh1557;  // [2,1]=+0.16670
         10'd658: kernel_real = 16'sh3E42;  // [2,2]=+0.48638
         10'd659: kernel_real = 16'shE3F5;  // [2,3]=-0.21908
         10'd660: kernel_real = 16'sh2E65;  // [2,4]=+0.36245
         10'd661: kernel_real = 16'shAD97;  // [2,5]=-0.64383
         10'd662: kernel_real = 16'sh0C9A;  // [2,6]=+0.09846
         10'd663: kernel_real = 16'shF2ED;  // [2,7]=-0.10214
         10'd664: kernel_real = 16'shD181;  // [3,0]=-0.36324
         10'd665: kernel_real = 16'shC5E4;  // [3,1]=-0.45399
         10'd666: kernel_real = 16'shE7F1;  // [3,2]=-0.18796
         10'd667: kernel_real = 16'sh1013;  // [3,3]=+0.12557
         10'd668: kernel_real = 16'shE0B4;  // [3,4]=-0.24451
         10'd669: kernel_real = 16'shE238;  // [3,5]=-0.23265
         10'd670: kernel_real = 16'shC36B;  // [3,6]=-0.47328
         10'd671: kernel_real = 16'shBD21;  // [3,7]=-0.52244
         10'd672: kernel_real = 16'shD181;  // [4,0]=-0.36324
         10'd673: kernel_real = 16'sh3A1C;  // [4,1]=+0.45399
         10'd674: kernel_real = 16'shE7F1;  // [4,2]=-0.18796
         10'd675: kernel_real = 16'shEFED;  // [4,3]=-0.12557
         10'd676: kernel_real = 16'shE0B4;  // [4,4]=-0.24451
         10'd677: kernel_real = 16'sh1DC8;  // [4,5]=+0.23265
         10'd678: kernel_real = 16'sh3C95;  // [4,6]=+0.47328
         10'd679: kernel_real = 16'shBD21;  // [4,7]=-0.52244
         10'd680: kernel_real = 16'shD35C;  // [5,0]=-0.34876
         10'd681: kernel_real = 16'shEAA9;  // [5,1]=-0.16670
         10'd682: kernel_real = 16'sh3E42;  // [5,2]=+0.48638
         10'd683: kernel_real = 16'sh1C0B;  // [5,3]=+0.21908
         10'd684: kernel_real = 16'sh2E65;  // [5,4]=+0.36245
         10'd685: kernel_real = 16'sh5269;  // [5,5]=+0.64383
         10'd686: kernel_real = 16'shF366;  // [5,6]=-0.09846
         10'd687: kernel_real = 16'shF2ED;  // [5,7]=-0.10214
         10'd688: kernel_real = 16'shD207;  // [6,0]=-0.35915
         10'd689: kernel_real = 16'shE626;  // [6,1]=-0.20196
         10'd690: kernel_real = 16'shC6F1;  // [6,2]=-0.44577
         10'd691: kernel_real = 16'sh41B8;  // [6,3]=+0.51343
         10'd692: kernel_real = 16'sh29A3;  // [6,4]=+0.32528
         10'd693: kernel_real = 16'shEAFC;  // [6,5]=-0.16420
         10'd694: kernel_real = 16'sh3490;  // [6,6]=+0.41066
         10'd695: kernel_real = 16'sh2102;  // [6,7]=+0.25786
         10'd696: kernel_real = 16'shD423;  // [7,0]=-0.34269
         10'd697: kernel_real = 16'sh3CC2;  // [7,1]=+0.47467
         10'd698: kernel_real = 16'sh15F1;  // [7,2]=+0.17143
         10'd699: kernel_real = 16'sh352F;  // [7,3]=+0.41548
         10'd700: kernel_real = 16'shC653;  // [7,4]=-0.45060
         10'd701: kernel_real = 16'shF785;  // [7,5]=-0.06626
         10'd702: kernel_real = 16'shD7FF;  // [7,6]=-0.31252
         10'd703: kernel_real = 16'sh3199;  // [7,7]=+0.38747
         // MODE 11: RFT-Entropy (unitarity: 1.80e-15)
         10'd704: kernel_real = 16'shC91C;  // [0,0]=-0.42882
         10'd705: kernel_real = 16'shF642;  // [0,1]=-0.07612
         10'd706: kernel_real = 16'sh4964;  // [0,2]=+0.57336
         10'd707: kernel_real = 16'shD8F9;  // [0,3]=-0.30492
         10'd708: kernel_real = 16'shEF7C;  // [0,4]=-0.12902
         10'd709: kernel_real = 16'shC5D6;  // [0,5]=-0.45441
         10'd710: kernel_real = 16'sh2CD3;  // [0,6]=+0.35020
         10'd711: kernel_real = 16'shE583;  // [0,7]=-0.20694
         10'd712: kernel_real = 16'shF944;  // [1,0]=-0.05261
         10'd713: kernel_real = 16'shC08D;  // [1,1]=-0.49570
         10'd714: kernel_real = 16'sh1069;  // [1,2]=+0.12819
         10'd715: kernel_real = 16'sh4E3E;  // [1,3]=+0.61127
         10'd716: kernel_real = 16'shFAF0;  // [1,4]=-0.03954
         10'd717: kernel_real = 16'shD34A;  // [1,5]=-0.34930
         10'd718: kernel_real = 16'shC792;  // [1,6]=-0.44086
         10'd719: kernel_real = 16'shE54E;  // [1,7]=-0.20854
         10'd720: kernel_real = 16'sh3A44;  // [2,0]=+0.45519
         10'd721: kernel_real = 16'shE528;  // [2,1]=-0.20972
         10'd722: kernel_real = 16'shCE28;  // [2,2]=-0.38941
         10'd723: kernel_real = 16'shF146;  // [2,3]=-0.11506
         10'd724: kernel_real = 16'shC22F;  // [2,4]=-0.48295
         10'd725: kernel_real = 16'shE473;  // [2,5]=-0.21524
         10'd726: kernel_real = 16'sh2A67;  // [2,6]=+0.33127
         10'd727: kernel_real = 16'shC788;  // [2,7]=-0.44118
         10'd728: kernel_real = 16'sh29B4;  // [3,0]=+0.32580
         10'd729: kernel_real = 16'sh39E2;  // [3,1]=+0.45222
         10'd730: kernel_real = 16'sh0739;  // [3,2]=+0.05644
         10'd731: kernel_real = 16'shEDD7;  // [3,3]=-0.14189
         10'd732: kernel_real = 16'sh3FD1;  // [3,4]=+0.49855
         10'd733: kernel_real = 16'shD2B7;  // [3,5]=-0.35379
         10'd734: kernel_real = 16'shDD5B;  // [3,6]=-0.27067
         10'd735: kernel_real = 16'shC418;  // [3,7]=-0.46802
         10'd736: kernel_real = 16'shD64C;  // [4,0]=-0.32580
         10'd737: kernel_real = 16'sh39E2;  // [4,1]=+0.45222
         10'd738: kernel_real = 16'sh0739;  // [4,2]=+0.05644
         10'd739: kernel_real = 16'sh1229;  // [4,3]=+0.14189
         10'd740: kernel_real = 16'shC02F;  // [4,4]=-0.49855
         10'd741: kernel_real = 16'sh2D49;  // [4,5]=+0.35379
         10'd742: kernel_real = 16'shDD5B;  // [4,6]=-0.27067
         10'd743: kernel_real = 16'shC418;  // [4,7]=-0.46802
         10'd744: kernel_real = 16'shC5BC;  // [5,0]=-0.45519
         10'd745: kernel_real = 16'shE528;  // [5,1]=-0.20972
         10'd746: kernel_real = 16'shCE28;  // [5,2]=-0.38941
         10'd747: kernel_real = 16'sh0EBA;  // [5,3]=+0.11506
         10'd748: kernel_real = 16'sh3DD1;  // [5,4]=+0.48295
         10'd749: kernel_real = 16'sh1B8D;  // [5,5]=+0.21524
         10'd750: kernel_real = 16'sh2A67;  // [5,6]=+0.33127
         10'd751: kernel_real = 16'shC788;  // [5,7]=-0.44118
         10'd752: kernel_real = 16'sh06BC;  // [6,0]=+0.05261
         10'd753: kernel_real = 16'shC08D;  // [6,1]=-0.49570
         10'd754: kernel_real = 16'sh1069;  // [6,2]=+0.12819
         10'd755: kernel_real = 16'shB1C2;  // [6,3]=-0.61127
         10'd756: kernel_real = 16'sh0510;  // [6,4]=+0.03954
         10'd757: kernel_real = 16'sh2CB6;  // [6,5]=+0.34930
         10'd758: kernel_real = 16'shC792;  // [6,6]=-0.44086
         10'd759: kernel_real = 16'shE54E;  // [6,7]=-0.20854
         10'd760: kernel_real = 16'sh36E4;  // [7,0]=+0.42882
         10'd761: kernel_real = 16'shF642;  // [7,1]=-0.07612
         10'd762: kernel_real = 16'sh4964;  // [7,2]=+0.57336
         10'd763: kernel_real = 16'sh2707;  // [7,3]=+0.30492
         10'd764: kernel_real = 16'sh1084;  // [7,4]=+0.12902
         10'd765: kernel_real = 16'sh3A2A;  // [7,5]=+0.45441
         10'd766: kernel_real = 16'sh2CD3;  // [7,6]=+0.35020
         10'd767: kernel_real = 16'shE583;  // [7,7]=-0.20694
         default:  kernel_real = 16'sh0000;
      endcase
   endfunction
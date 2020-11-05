import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl 

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
#    "text.usetex": True,     # use inline math for ticks
#    "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

import sys
import glob
import scipy.stats 
from io import StringIO
plt.style.use('ggplot')

import ruamel.yaml as yaml 
import subprocess

TXDLY = 15e-6
ANTDLY = 15e-6

# From the GDP manual, section 12.9. 
# these apply to 32, 16, 8 and 4 Hz for centre 
# values, others can be calculated. 
WINTBL = """
1   1   0.0u    0.0u    0.0u
2   1   30.5u   30.5u   30.5u 
3   1   61.0u   61.0u   61.0u
4   1   91.6u   91.6u   91.6u
5   1   122.1u  122.1u  122.1u
6   1   152.6u  152.6u  152.6u
7   2   197.8u  183.1u  213.6u
8   2   259.0u  244.1u  274.7u
9   2   320.1u  305.2u  335.7u
10  3   395.6u  366.2u  427.3u
11  3   487.3u  457.8u  518.8u
12  5   607.3u  549.3u  671.4u
13  6   774.5u  701.9u  854.5u
14  7   972.3u  885.0u  1.068m
15  9   1.215m  1.099m  1.343m
16  11  1.518m  1.373m  1.678m
17  15  1.911m  1.709m  2.136m
18  19  2.426m  2.167m  2.716m
19  23  3.064m  2.747m  3.418m
20  29  3.852m  3.449m  4.303m
21  36  4.838m  4.334m  5.402m
22  47  6.094m  5.432m  6.836m     
23  58  7.687m  6.867m  8.606m
24  72  9.659m  8.637m  10.803m
25  92  12.14m  10.834m 13.611m    
26  116 15.30m  13.642m 17.151m
27  145 19.25m  17.182m 21.576m
28  184 24.24m  21.607m 27.192m    
29  231 30.53m  27.222m 34.241m
30  289 38.42m  34.272m 43.061m
31  369 48.38m  43.091m 54.322m   
"""

def convert(number):
    """ 
    Converts GDP data which may contain scaling factors into floats
    
    Args: 
        number (string) : String representation of a number, eg. 1.234u which is 
                          converted to 1.234e-6

    Returns:
        float: The converted number in floating point precision. 
 
    """
    if number[-1].isalpha():
        sc = number[-1]
        if sc == 'M':
            return float(number[0:-1])*1e6
        elif sc == 'K':
            return float(number[0:-1])*1e3
        elif sc == 'm':    
            return float(number[0:-1])*1e-3
        elif sc == 'u':
            return float(number[0:-1])*1e-6
        elif sc == 'n':
            return float(number[0:-1])*1e-9
    else:
        # default case of no scaling factor         
        return float(number)

def extractWindowTuples(WIN):
    """Convenience function that converts a window matrix to tuples of 
       windows for a particular record.
    """
    wt = []
    for iw in range(len(WIN)):
        wt.append( np.array( [WIN[iw][1], WIN[iw][2]] ))
    return np.array(wt)


class TEMSounding(  ):
    """ 
    SuperClass for TEM soundings
    """

    def __init__( self ):
        self.nStack = 0
        self.nTimeGates = 0

class ZeroTEMSounding( TEMSounding ):
    """
    A ZeroTEM sounding as recorded by Zonge instrumentation. 
    """

    def __init__(self):
        super(ZeroTEMSounding, self).__init__()
        self.stacks = []

    def loadStack(self, stackDir):
        """
        Loads a directory of stacks, each stack in the directory is assumed to have the same 
        parameters including sampling frequency and current. If a series of stacks has already been 
        loaded, they will be replaced by this record.  
        
        Args:
            stackDir : Directory path containing the stacks
        """
        print(stackDir)

        self.WN, self.MAG, self.RHO  = [],[],[]
        for SND in glob.glob(stackDir+"/*.SND"):
            self.stacks.append(SND)
            wn, mag, rho, WIN = self.loadSND(SND)
            self.WN.append(wn)
            self.MAG.append(mag)
            self.RHO.append(rho)
            self.nStack += 1

        self.WIN = WIN # all windows **should** be aligned, TODO be more careful
        self.WN = np.array(self.WN).T
        self.MAG = np.array(self.MAG).T
        self.RHO = np.array(self.RHO).T
        print ("Loaded", self.nStack, "soundings in", stackDir)

    def plotStack(self, freq, site):
        """
        Plots the stacked and averaged data
        """
        global firstPlot
        global ax1
        global ax2
 
        fig = plt.figure(0, figsize=(7.0,6.0))
        if firstPlot:
            ax1 = fig.add_axes([.15,.300,.8,.65])
            ax2 = fig.add_axes([.15,.100,.8,.15], sharex=ax1)
            #ax2 = fig.add_axes([.15,.15,.75,.75], sharex=ax1)


        # calculate average 
        self.AVG = np.average(self.MAG, axis=1)

        # go ahead and fix sign errors
        if self.AVG[0] < 0:
            self.AVG *= -1.
            self.MAG *= -1.
            
        neg = self.MAG <=0 
        pos = self.MAG > 0 
        
        ax1.plot(self.WN[neg],   self.MAG[neg], '_', alpha=.25, color=colours[isnd]) # grey
        ax1.plot(self.WN[pos],    self.MAG[pos], '+', alpha=.25, color=colours[isnd])

        #ax1.plot(self.WN, self.MAG, 'o', alpha=.15, markersize=3, color=colours[isnd]) # grey

        if np.shape(self.MAG)[1] > 1:
            self.STD = np.std( self.MAG, axis=1 )
        else:
            self.STD = 1e-5*np.ones( len(self.AVG) )
            #self.STD = np.std(self.AVG[-8::]) * np.ones(len(self.AVG))
            #print("assigning dummy variance", self.STD)
        #self.STD[self.STD<1e-5] += 5e-4
        
        neg = self.AVG <=0 
        pos = self.AVG > 0 

        self.mask = np.abs(self.AVG) < 1. * self.STD     
        if self.STD[0] < 1e-7:
        #    self.mask[0] = True
            self.STD[0] += self.STD[1]

        self.mask[0:1] = True
        #self.mask[3:] = True
 
        #ax1.plot(self.WN, self.mask, 'o', alpha=.25, markersize=8, color='black') 

 
        # average apparent resistivity 
        AVGR = np.average(self.RHO, axis=1)
        
        # simple average
        #plt.scatter(self.WN[neg,0], -1*AVG[neg], marker='_', color = colours[isnd], alpha=1, s=80)
        #plt.scatter(self.WN[pos,0],    AVG[pos], marker='+', color = colours[isnd], alpha=1, s=80)
        #plt.plot(self.WN[:,0], np.abs(AVG), '-', color=colours[isnd], alpha=1, label=freq)
        
        ax1.plot(self.WN[:,0], self.AVG, '-', color=colours[isnd], alpha=1, linewidth=1) 
        ax1.errorbar(self.WN[:,0], self.AVG, yerr=self.STD, fmt='o', markersize=4, markeredgecolor='black', markeredgewidth=.5, color=colours[isnd], alpha=1, label=freq)

        if firstPlot:
            ax1.plot(self.WN[self.mask,0], self.AVG[self.mask], 's', color='black', alpha=1, markersize=5, label='masked') 
        else:
            ax1.plot(self.WN[self.mask,0], self.AVG[self.mask], 's', color='black', alpha=1, markersize=5) 

        # SNR 
        ax2.plot(self.WN[:,0], 20*np.log10(np.abs(self.AVG/(self.STD))), '.-', color=colours[isnd], alpha=1, label=freq)
        ax2.axhline(y=1, linestyle='--', color='black')
 
        # reject above 2 STD
        if False:
            #STD = np.std( self.MAG, axis=1 )
            OUT = self.MAG-np.tile(self.AVG, (self.nStack,1)).T > 2.*np.tile(self.STD, (self.nStack,1)).T
            print("Removed", np.sum(OUT), "outliers")
            self.AVG = np.ma.masked_array(self.MAG, OUT==True).mean(axis=1) 
            #plt.plot(self.WN[:,0], AVG2, '-', color=colours[isnd+1], alpha=1, label=freq)

        if False:
        # MAD outlier detection
            MAD = scipy.stats.median_abs_deviation( self.MAG, axis=1, scale="normal" )
            MED = np.tile(np.median(self.MAG, axis=1), (self.nStack,1)).T
            OUT = ( np.abs(self.MAG-MED) / np.tile(MAD, (self.nStack,1)).T  ) > 2
            print("Removed", np.sum(OUT), "outliers")
            self.AVG = np.ma.masked_array(self.MAG, OUT==True).mean(axis=1) 
            #plt.plot(self.WN[:,0], AVG2, '-', color=colours[isnd+1], alpha=1, label=freq)
        
        ax1.set_yscale('symlog', linthresh=self.STD[-1:])
        ax2.set_yscale('symlog') #, linthresh=1e-6)
        #ax2.set_yscale('log')
        #ax1.xaxis.set_ticklabels([])


        ax1.set_xscale('log')
        #ax1.set_xscale('symlog', linthresh=1e-5)
        #ax2.set_xscale('log')
        ax2.set_xlabel("time (s)")
        ax1.set_ylabel("$\dot{H}_z$ (V)")
        ax2.set_ylabel("S:N (dB)")
        ax1.set_title(site)

        # don't show tick labels on top plot
        ax1.xaxis.set_tick_params(which='both', labelbottom=False)

        ax1.legend( )

        global pgfTitle
        try: 
            pgfTitle += "_" + str(freq)
        except:
            pgfTitle = str(freq)

        #plt.savefig(pgfTitle+"_stack.pgf")
        plt.savefig(pgfTitle+"_stack.pdf")

        # apparent resistivity plot
        if False:
            plt.figure(1, figsize=[3,4])
            plt.plot( self.WN[:,0], AVGR )
            plt.title("resistivity")      
            plt.gca().set_ylabel("Apparent resistivity ($\Omega \cdot \mathrm{m}$)") 
            #plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.gca().set_xlabel("time (s)")
            plt.savefig(pgfTitle+"_ar.pdf")


        firstPlot = False

        # write out Beowulf inversion filess
        #writeCFL("Beowulf.cfl")
        #writeINV("Beowulf.inv", MAG)

    def loadSND(self,filename):
        """ 
        Loads a sounding saved by a GDP 

        Args:
            filename(string) : the filename to load          

        """
        inp = open(filename, 'r')
        lines = inp.readlines()
        inp.close()

        header = []
        for hl in range(6):
            header.append(lines[hl])

        WIN = self.calculateWindows(header)

        wn, mag, rho = [],[], []
        for fl in range(6, len(lines)):
            parse = (lines[fl].split())
            if len(parse) > 0:
                wn.append(convert(parse[0]))
                mag.append(convert(parse[1]))
                rho.append(convert(parse[2]))
        
        if len(wn) != self.nTimeGates:
            if self.nTimeGates == 0: 
                self.nTimeGates = len(wn)
            else:
                # TODO consider an exception here
                print("Attempt to stack non-aligned SND files")
                exit()

        return(np.array(wn),np.array(mag),np.array(rho),WIN)

    def Window(self, hz, offset):
        """ 
        Determintes the windows based on Zonge table data which are a function 
        of Tx, Rx, and sampling delay.   
        """
        sc = 1.
        if hz < 4:
            sc = 4./hz
        return np.genfromtxt(StringIO(WINTBL), dtype= [('WN','i8'),('NP','i8'),('win_centre','f8'), \
            ('win_beg','f8'),('win_end','f8')], comments="#", converters=\
                {2: lambda s: convert(s.decode('utf-8'))*sc + offset ,\
                 3: lambda s: convert(s.decode('utf-8'))*sc + offset ,\
                 4: lambda s: convert(s.decode('utf-8'))*sc + offset })

    def calculateWindows(self, header):
        """ 
        Parses the header data of a GDP record in order to extract window information, 
        Uses Window function to do this as well. 
        """
        recordNumber = header[0]  # just the GDP record index 
        # Parse second line
        stype, gpdN, date, time, batt, rtype, humidity, temp, tunits = header[1].split()
        # Parse third line 
        Tx, nRx, Rx, line, NSEW, out = header[2].split()
        # parse 4th line
        Freq, Hz, Ncycles, Cyc, Tx, Curr, Amps, sampDly, aAliasDly, offset = header[3].split()
        sampDly = convert(sampDly)
        aAliasDly = convert(aAliasDly)
        offset = convert(offset)

        self.Freq = Freq
        self.sampDly = sampDly

        # save 
        self.sampFreq = float(Freq)  
        self.Amps = Amps 

        # the time gates are adjusted for low frequency datasets 
        sc = 1.
        if float(Freq) < 4:
            sc = 4./float(Freq)

        # Calculate 1st time gate according to manual, seems consistent with table 
        First = sampDly - (TXDLY+ANTDLY+aAliasDly)

        # 2e-7 is an ad hoc correction that seems to work across datasets 
        SAMPLING = np.arange(1,2000)*(offset - sc*2e-7) 
  
        # grab data from table, TODO can probably just grab the single column we need
        samp = self.Window(float(Freq), offset) #-(TXDLY+ANTDLY))

        wc,wb,we,ii = [First],[First],[First],[1]
        sStart = 0  # always starts at a 1 sample at First
        for iw in range(1,len(samp)):
            sEnd = sStart + samp[iw][1]
        
            wc.append( First + np.mean( SAMPLING[sStart:sEnd] ) )
            wb.append( First + SAMPLING[sStart] )
            we.append( First + SAMPLING[sEnd-1] )
            ii.append( samp[iw][1]  )
            sStart = sEnd 
    
        return np.array( (wc,wb,we,ii) ).T
    
    def export(self, sdir):

        self.writeCFL("Beowulf.cfl", sdir)
        self.writeINV("Beowulf.inv", np.average(self.MAG, axis=1))

    def writeCFL(self, fname, sdir):
        
        try:
            CTRL = yaml.load(open(sdir+'/control.yaml'), Loader=yaml.Loader)
        except:
            print("No Control file found! Cannot export CFL for inversion")
            exit(1)

        if CTRL["Invert"] == "False":
            return

        # dummy waveform base length on 
        #TXAMP = [0., 4.5, 4.5, 0.5, 0.]
        #TXABS = [0, 1.5, 1e3*0.5/self.sampFreq-0.125, 1e3*0.5/self.sampFreq] # Good for 1 Hz 
        #TXABS = [0, 1.5,  \
        #    1e3*0.5/self.sampFreq-0.025, \
        #    1e3*0.5/self.sampFreq-0.015, \
        #    1e3*0.5/self.sampFreq]

        TXABS = CTRL['TxAbs'] 
        TXAMP = CTRL['TxAmp'] 

#         if self.Freq == "1":
#             ############################################################
#             # 1 Hz from WFM file 
#             TXABS = np.array([  0.   ,   0.42 , 249.958, 250.09 ])     #
#             TXAMP = [0.,4.25,4.25,0]                       # wfm file  #
#             ############################################################
#         elif self.Freq == "8":
#             ############################################################
#             # 8 Hz from WFM file 
#             TXABS = np.array([ 0.  ,  0.5 , 31.22, 31.35]) # 8 Hz from #
#             TXAMP = [0.,4.25,4.25,0]                       # wfm file  #
#             #TXABS = np.array([ 0.  ,  0.5 , 31.22, 31.249]) # 8 Hz from #
#             #TXAMP = [0.,4.25,4.25,0]                       # wfm file  #
#             ############################################################ 
#         elif self.Freq == "16":
#             ############################################################
#             # 16 Hz from WFM file 
#             TXABS = [0.,0.05,0.75,15.65,15.75]  # 16 Hz, from WFM file #
#             TXAMP = [0.,2.5,4.25,4.25,0]                               #
#             ############################################################ 
#         elif self.Freq == "32":
#             ############################################################
#             # 16 Hz from WFM file 
#             TXABS = [0.,0.495,7.809,7.92]  # 32 Hz, from WFM file #
#             TXAMP = [0.,4.25,4.25,0]                                   #
#             ############################################################ 

        # Good for 1 Hz 
        #TXABS = [0, 1.5, 1e3*0.5/self.sampFreq-0.025, 1e3*0.5/self.sampFreq] # Good for 16 Hz 
        #TXABS = [-1e3*0.5/self.sampFreq, -1e3*0.5/self.sampFreq+1.5, 0, 1.5]  # TODO look at WFM files 

        cw = open(fname, "w")
        # RECORD 1
        cw.write("Flathead inversion\n")
        ############
        # RECORD 2 #
        ############
        cw.write("1 0 0                      !TDFD, ISYS, ISTOP\n")     
        ############
        # RECORD 3 #
        ############
        # Step 0 = db/dt
        # NSX = number of points to describe waveform 
        # NCHNL = number of time gates 
        # KRXW = time gates described as start to end (1) or centre and width (2)
        # REFTYM = Zero time for data 
        # OFFTIME = time between cycles in ms  
        cw.write("0 %2.i " %len(TXAMP)) 
        #cw.write( str(self.nTimeGates) + " 2 " ) # n time gates, KRXW 
        print("Number of non-masked time gates", np.sum(self.mask==0)) 
        cw.write( str(np.sum(self.mask==0)) + " 2 " ) # n time gates, KRXW  
        # REFTYM : GDP starts the clock at the start of the ramp off...I THINK TODO verify 
        # OFFTIME is not saved explicitly in GDP files, this is a rough approximation...
        #cw.write(" 0.0   %2.4f " %(.5/self.sampFreq))
        #cw.write(" %2.4f   %2.4f " %(TXABS[-2], 1e3*(.5/self.sampFreq)) )
        
        #############################################################################
        #cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.039, TXABS[-1])) # 16 Hz           #
        #cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.03855, TXABS[-1]))                 #
        #cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.03863, TXABS[-1])) # From header   #
        cw.write(" %2.4f   %2.4f " %(TXABS[-1] - CTRL['TxDly'], TXABS[-1])) 
        
#         if self.Freq == "1":
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.10, TXABS[-1])) 
#             cw.write(" %2.4f   %2.4f " %(TXABS[-1], TXABS[-1])) 
#         elif self.Freq == "8":
#             cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.039, TXABS[-1])) # 10.04 RMS, S4-1 
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-2]+.125, TXABS[-1])) # 10.04 RMS, S4-1 
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-2], TXABS[-1])) # 10.04 RMS, S4-1 
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-1], TXABS[-1])) # 10.04 RMS, S4-1 
#         elif self.Freq == "16":
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.03863, TXABS[-1])) # 19.07 RMS, S3B  
#             cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.038, TXABS[-1])) # 19.07 RMS, S3B  
#             #cw.write(" %2.4f   %2.4f " %(TXABS[-1], TXABS[-1])) 
#         elif self.Freq == "32":
#             cw.write(" %2.4f   %2.4f " %(TXABS[-1]-.040, TXABS[-1])) # 11.85 RMS, S4-1 


        #############################################################################

        #cw.write(" %2.4f   %2.4f " %(TXABS[-2], 1e3*(.5/self.sampFreq)) )
        cw.write( "   !STEP, NSX, NCHNL, KRXW, REFTYM, OFFTIME")
       
        for ii in range(len(TXAMP)):
            cw.write("\n%8.3f %7.3f" %(TXABS[ii], TXAMP[ii]))
        cw.write("           !Tx Wvfm: abscissa (ms), current (A)\n") 
        
        # write out data gates...
        width = self.WIN[:,2] - self.WIN[:,1]
        width1 = (self.WIN[1,0]-self.WIN[0,0]) / 2
        width[width<width1/2] = width1 # I'm not sure if zero length widths are allowed

        # KRXW==1 start and off time 
        #for tg in range(self.nTimeGates):
        #    cw.write( "%9.4f %7.4f\n"   %((round(1e3*self.WN[tg,0],5)),(round(1e3*width[tg],5))))
        
        # KRXW==2 requires these to follow each other 
        for tg in range(self.nTimeGates):
            if self.mask[tg] == False:
                cw.write( "%9.4f\n"   %((round(1e3*self.WN[tg,0],5))))
        for tg in range(self.nTimeGates):
            if self.mask[tg] == False:
                cw.write( "%9.4f\n"   %((round(1e3*width[tg],5))))
       
        cw.write("1                          !SURVEY_TYPE: General\n")

        # Transmitter  
        TxType = CTRL["TxType"] 
        TxSz = CTRL["TxSize"]

        #TxType = "Circle" 
        if TxType == "Circle":
            print("Using 55m CIRCULAR transmitter")
            rad = TxSz
            TXX = rad*np.sin(np.linspace(0,2*np.pi,32, endpoint=False))
            TXY = rad*np.cos(np.linspace(0,2*np.pi,32, endpoint=False))
        else:
            print("Using 100m SQUARE transmitter")
            TXX = np.array( [-TxSz/2,  TxSz/2, TxSz/2, -TxSz/2] )
            TXY = np.array( [-TxSz/2, -TxSz/2, TxSz/2,  TxSz/2] )
        #plt.figure()
        #plt.plot(TXX, TXY)
        #plt.show()

        cw.write("1 1 1 1 50 1               !NLINES, MRXL, NTX, SOURCE_TYPE, MAXVRTX, NTURNS\n") 
        cw.write("%2.i 0                     !Nvertex, elevation z" %(len(TXX)))
        for i in range(len(TXY)):
            cw.write( "\n%8.3f %8.3f" %(TXY[i],TXX[i]) )
        cw.write( "          !Tx[i] East, Tx North (m)\n")

        # specify rx 
        #LINE(J), IDTX(J), RX_TYPE(J), NRX(J), UNITS(J)
        cw.write( "1000 1 1 1 1              !Line txid, rxtype, nrx, units (V)\n")
        # units 11 = nT/s 

        #CMP(J), SV_AZM(J),KNORM(J), IPLT(J), IDH(J), RXMNT(J)
        cw.write("3 0 0 1 0 10000            !cmp, sv_azm, knorm, iplt, idh, rxmoment\n")
        cw.write("0 0 0                      !receiver position\n")

        # Record 10
        SMOOTH = CTRL['Smooth']
        nilay = CTRL['Nlay']

        if len(CTRL['SRes']) == 1:
            res =  CTRL['SRes']*np.ones(nilay) 
        else:
            res = CTRL["SRes"]        


        if SMOOTH:
            ##########################
            # SMOOTH INVERSION 
            ##########################
            cw.write("%i %i                        ! NLAYER, NLITH\n" %(nilay, nilay))
            thick = np.geomspace(3, 60, nilay)
            #res = 20*np.ones(nilay) #np.array([26, 12., 87, 187, 307, 364, 237, 28, 2565])
            for ii in range(nilay):
                cw.write("%2.2f  1 1 0 0 0                ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n" %(res[ii]))
            for ii in range(nilay):
                cw.write("%i %i                          ! LITH, THICK - Layer 1\n" %(ii+1, thick[ii]))
        else:
            ##########################
            # MINIMUM LAYER INVERSION 
            ##########################
            cw.write("%i %i                        ! NLAYER, NLITH\n" %(nilay, nilay))
            thick = np.geomspace(5, 100, nilay)
            #res = 20*np.ones(nilay) #np.array([26, 12., 87, 187, 307, 364, 237, 28, 2565])
            #res[0] = .01
            for ii in range(nilay):
                cw.write("%2.2f  1 1 0 0 0                ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n" %(res[ii]))
            for ii in range(nilay):
                cw.write("%i %i                          ! LITH, THICK - Layer 1\n" %(ii+1, thick[ii]))
 
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("200.6  1 1 0 0 0             ! RES, RMU, REPS, CHRG, CTAU, CFREQ(1) - Lyr1\n")
        #cw.write("1 5                          ! LITH, THICK - Layer 1\n")
        #cw.write("2 10                         ! LITH, THICK - Layer 2\n")
        #cw.write("3 20                         ! LITH, THICK - Layer 2\n")
        #cw.write("4 40                         ! LITH, THICK - Layer 2\n")
        #cw.write("5 65                         ! LITH, THICK - Layer 2\n")
        #cw.write("6 25                         ! LITH, THICK - Layer 2\n")
        #cw.write("7 25                         ! LITH, THICK - Layer 2\n")
        #cw.write("8 25                         ! LITH, THICK - Layer 2\n")
        #cw.write("9 25                         ! LITH, THICK - Layer 2\n")
        #cw.write("10                           ! LITH - basement\n")
        
        if SMOOTH:
            cw.write("%i 90 10 2                    ! NFIX, MAXITS, CNVRG, INVPRT\n" %(nilay-1))
        else:
            cw.write("0 90 1 2                    ! NFIX, MAXITS, CNVRG, INVPRT\n") 
        
        # write out std err W CNVRG = 2
        #std = np.average(self.MAG, axis=1) 
        #for ii in range(len(self.STD[self.mask!=1]) ): 
        #    cw.write( str(self.STD[self.mask!=1][ii] ) + "\n")
        #cw.write(str(.9) + "\n")
        
        # derivative search 
        cw.write("     3         ! NDSTP\n")
        cw.write("     5 3 1     ! KPCT (1:NDSTP)\n")

        if SMOOTH:
            # fixed layer thickness 
            for ilay in range(1, nilay):
                cw.write("1 " + str(ilay) + " 2\n")

        cw.close()


    def writeINV(self, fname, data):
        bw = open(fname, "w")
    
        bw.write("0          ! FD_ORDER\n") # time domain 
    
        # LINE_CHK FID for consistency with cfl file
        # NSTAT is the number of receivers / stations
        # KMP is the component, 3 == z component  
        bw.write("1000  1 3  ! LINE_CHK, NSTAT, KMP(J)\n") 
    
        bw.write("0      ! DATA_FLOOR(J)\n") # 16
    
        # write out the data now 
        bw.write("      1") # leading white space is in AMIRA example files?, Receiver channel index  

        sc = 1.   
        if data[0] < 0:
            sc = -1.
 
        for ii in range(len(data)):
            #bw.write("      " + str( round(data[ii] * 1e4, 5)  ))
            if self.mask[ii] == False:
                bw.write("      " + str(sc*1e-4*data[ii]) ) # rx moment
        bw.write("\n")
        bw.close() 

    def invert(self, sdir):
        """ Calls AMIRA Beowulf inversion. 
        """
        subprocess.call("./Beowulf")
        subprocess.call(["copy", "Beowulf.mv1", sdir+".mv1"])

    def readMV1(self, filename):
        with open(filename) as mv1:
            print("opening MV1", filename)
            for line in mv1:
                lsplit = line.split()
                if len(lsplit) > 1 and lsplit[1] == "TIMES(ms)=":
                    tg = np.array(lsplit[2:], dtype=float)
                if len(lsplit) > 1 and lsplit[1][0:7] == "LAYERS=":
                    #nlay = np.array(lsplit[1][8:], dtype=int)
                    nlay = np.array(lsplit[1].split("=")[1], dtype=int)
                if len(lsplit) > 1 and lsplit[1] == "FINAL_MODEL":
                    sig = np.array( lsplit[2:nlay+2], dtype=float )
                    thick = np.array( lsplit[nlay+2:], dtype=float )
                if len(lsplit) > 1 and lsplit[0] == "SURVEY_DATA":
                    obs = np.array(lsplit[4:], dtype=float)
                if len(lsplit) > 1 and lsplit[0] == "MODEL_DATA":
                    pre = np.array(lsplit[4:], dtype=float)

            # make bottom layer thick...
            thick = np.concatenate( [thick, [500]] )
            depth = np.concatenate( [[0], np.cumsum(thick)] )
            depthc =  (depth[0:-1] + depth[1:] ) / 2

        return sig, thick, depth, depthc, obs, pre, tg

    def modelAppraisal(self, sdir):

        sig, thick, depth, depthc, obs, pre, tg = self.readMV1(sdir+".mv1")
        tg *= 1e-3

        #subprocess.call(["python", "plotMV1.py", sdir+".mv1"])
        #print("std", self.STD[self.mask!=1])
        #print("obs - pre", np.abs(obs-pre))

        print( "L2 norm=", np.linalg.norm( 1e4*(obs-pre) / (self.STD[self.mask!=1]) ))

        # Spies DOI estimate
        CTRL = yaml.load(open(sdir+'/control.yaml'), Loader=yaml.Loader)
        beta = self.STD[self.mask!=1][-1]
        if CTRL["TxType"] == "Square":
            area = CTRL["TxSize"]**2
        else:
            area = np.pi*(CTRL["TxSize"]**2)
        I = CTRL["TxAmp"][-2]
        rhoa = np.sum(np.dot(sig,thick)) / np.sum(thick) # ohm m^2 

        DOI = 0.55 * ((I * area * rhoa ) / beta)**.2    

        print("rho_a", rhoa)
        print("beta", beta)
        print("DOI", DOI) 


        figa = plt.figure(3, figsize=(7,4.5))
        figa.clf()

        figa.suptitle(sys.argv[-1] + " " + sdir + " inversion result", fontsize=16)

        aax1 = figa.add_axes([.125,.45,.700,.40])
        aax2 = figa.add_axes([.850,.45,.025,.40])
        wmap = cm.get_cmap("viridis", 10)
    

        aax1.set_title("Recovered model")

        aax3 = figa.add_axes([.125,.1,.80,.15])
        #aax4 = figa.add_axes([.125,.15,.75,.325], sharex=aax3)
        #aax3.set_yscale('log')
        aax3.set_xscale('log')
        
        aax3.set_title("Data fit")

        #nnorm = mpl.colors.LogNorm(vmin=np.min(sig), vmax=np.max(sig)) 
        nnorm = mpl.colors.LogNorm(vmin=1, vmax=1e3) 
        sigc = wmap( nnorm( sig ) ) # mpl.colors.LogNorm(vmin=np.min(sig), vmax=np.max(sig))) #same as above 
        aax1.barh(depthc, width=sig, height=thick, color=sigc, alpha=1.)#color=wmap.colors)
    
        aax3.plot(self.WN,          self.MAG, 'o', alpha=.15, markersize=3, color=colours[isnd]) #, label="Observed") # grey
        aax3.errorbar(self.WN[:,0], self.AVG, yerr=self.STD, fmt='o', markersize=4, \
            markeredgecolor='black', markeredgewidth=.5, color=colours[isnd], alpha=1)

        aax3.plot(tg, 1e4*obs, '.', color=colours[isnd], label="Observed")
        aax3.plot(tg, 1e4*pre, '--', color='black', label="Predicted")

        aax1.text(30, DOI - 10, "DOI (Spies, 1989)", fontsize=10)
        aax1.axhline(DOI, color='black', linestyle='--')
        
        #aax4.plot(tg, (obs-pre)/(1e-4*self.STD[self.mask!=1]), '.-',label="misfit")


        aax3.set_xlabel("time (s)")
        aax3.set_ylabel("$\dot{H}_z$ (V)")

        #aax1.set_ylim( [DOI + 50,0]  )
        aax1.set_ylim( [CTRL["PDepth"], 0]  )
        #aax1.set_ylim( aax1.get_ylim()[::-1]  )
        aax1.set_xlim( [1,1e4]  )
        aax1.set_xscale('symlog')  # LaTeX complains unless this
        #aax1.set_yscale('symlog')

        aax1.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())  # only works for res > 1 

        #plt.colorbar(sigc)
        aax1.set_xlabel("resistivity ($\Omega \cdot \mathrm{m}$)", fontsize=12)
        aax1.set_ylabel("depth ($\mathrm{m}$)", fontsize=12)

        cb1 = mpl.colorbar.ColorbarBase(aax2, cmap=wmap,
                                   norm=nnorm,
                                   orientation='vertical', 
                                   extend='both')
        cb1.set_label("resistivity ($\Omega \cdot \mathrm{m}$)", fontsize=12)
        aax3.legend()
        figa.savefig( sdir+"_inv.pdf" )


if __name__ == "__main__":


    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #for SND in sys.argv[1:]:
    isnd = 0
    MAG = [] 

    firstPlot = True

    for sdir in sys.argv[1:-1]:

        # New class interface 
        ZT = ZeroTEMSounding()
        ZT.loadStack(sdir) # + "./*.SND")
        ZT.plotStack(sdir, sys.argv[-1])
            
        CTRL = yaml.load(open(sdir+'/control.yaml'), Loader=yaml.Loader)
        if CTRL["Invert"] != "False":
            ZT.export(sdir)

            # Call inversion
            ZT.invert(sdir)
            ZT.modelAppraisal(sdir)
        isnd += 1
    
    plt.show()

#         for SND in glob.glob(sdir+"./*.SND"):
#             
#             wn, mag, rho, WIN = loadSND(SND)
#             wint= extractWindowTuples(WIN)
#             neg = mag<=0
#             pos = mag>0
#             MAG.append(mag)
#             plt.scatter(wn[neg], -1*mag[neg], marker='.', color = colours[isnd], alpha=.25)
#             plt.scatter(wn[pos],    mag[pos], marker='+', color = colours[isnd], alpha=.25)
#             plt.plot( wiwintnt[0:len(wn)].T, np.ones((len(wn), 2)).T, color='grey' )
#             plt.plot( np.average(wint[0:len(wn)], axis=1).T, np.ones((len(wn))).T, '.', color='black',  )
#             #plt.plot( wint[0:len(wn)].T, np.ones((len(wn), 2)).T )
# 
#         # calculate average 
#         MAG = np.average(MAG, axis=0)
# 
#         # write out Beowulf inversion filess
#         #writeCFL("Beowulf.cfl")
#         #writeINV("Beowulf.inv", MAG)
# 
#         plt.scatter(wn[neg], -1*MAG[neg], marker='_', color = colours[isnd], alpha=1, s=80)
#         plt.scatter(wn[pos],    MAG[pos], marker='+', color = colours[isnd], alpha=1, s=80)
#         plt.plot(wn, np.abs(MAG), color = colours[isnd], alpha=1, label=sdir)
#   
#         isnd += 1 
#     
#     plt.gca().set_yscale('log')
#     plt.gca().set_xscale('log')
#     plt.legend()
# 
#     plt.show()

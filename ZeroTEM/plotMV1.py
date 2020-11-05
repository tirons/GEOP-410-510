import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import sys

plt.style.use('ggplot')

def plotModel(files):
    
    fig = plt.figure(1, figsize=(4,4))
    ax1 = fig.add_axes([.2,.40,.7,.5])
    ax2 = fig.add_axes([.2,.15,.7,.04])
    wmap = cm.get_cmap("viridis", 10)
    
    fig2 = plt.figure(2)
    ax3 = fig2.add_axes([.125,.35,.75,.5])
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    for filename in files:
        with open(filename) as mv1:
            print("opening", filename)
            for line in mv1:
                lsplit = line.split()
                if len(lsplit) > 1 and lsplit[1] == "TIMES(ms)=":
                    tg = np.array(lsplit[2:], dtype=float)
                if len(lsplit) > 1 and lsplit[1][0:7] == "LAYERS=":
                    nlay = np.array(lsplit[1][8:], dtype=int)
                if len(lsplit) > 1 and lsplit[1] == "FINAL_MODEL":
                    sig = np.array( lsplit[2:nlay+2], dtype=float )
                    thick = np.array( lsplit[nlay+2:], dtype=float )
                if len(lsplit) > 1 and lsplit[0] == "SURVEY_DATA":
                    obs = np.array(lsplit[4:], dtype=float)
                if len(lsplit) > 1 and lsplit[0] == "MODEL_DATA":
                    pre = np.array(lsplit[4:], dtype=float)

            # make bottom layer thick...
            thick = np.concatenate( [thick, [100]] )
            depth = np.concatenate( [[0], np.cumsum(thick)] )
            depthc =  (depth[0:-1] + depth[1:] ) / 2

            #nnorm = mpl.colors.LogNorm(vmin=np.min(sig), vmax=np.max(sig)) 
            nnorm = mpl.colors.LogNorm(vmin=1, vmax=1e3) 
            sigc = wmap( nnorm( sig ) ) # mpl.colors.LogNorm(vmin=np.min(sig), vmax=np.max(sig))) #same as above 
            ax1.barh(depthc, width=sig, height=thick, color=sigc, alpha=1.)#color=wmap.colors)
    
            ax3.plot(tg, obs, '.-', label="obs")
            ax3.plot(tg, pre, '.-',label="pre")

            ax3.set_xlabel("time (s)")
            ax3.set_ylabel("$\dot{H}_z$ (V)")


    #plt.fill_betweenx( depth, sig )
    #ax1.set_ylim( ax1.get_ylim()[::-1]  )
    ax1.set_ylim( [250,0]  )
    ax1.set_xlim( [1,1e3]  )
    
    ax1.set_xscale('symlog')

    #plt.colorbar(sigc)
    ax1.set_xlabel("resistivity ($\Omega \cdot \mathrm{m}$)", fontsize=12)
    ax1.set_ylabel("depth [$\mathrm{m}$]", fontsize=12)

    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=wmap,
                                   norm=nnorm,
                                   orientation='horizontal', 
                                   extend='both')
    cb1.set_label("resistivity ($\Omega \cdot \mathrm{m}$)", fontsize=12)
    print ("saving", sys.argv[1], "as", sys.argv[1][0:-3])
    fig.savefig( sys.argv[1][0:-4]+"_inv.pgf" )
    
    ax3.legend()

plotModel(sys.argv[1:])
plt.show()

# Python HDF5 snapshot reader for Gadget2
# (requirest PyTables http://www.pytables.org)
#
# import readsnapGadget2 as snap
# header = snap.snapshot_header("snap_063") 
# mass = snap.read_block("snap_063", "MASS", parttype=1) 
#
# Mark Vogelsberger (mvogelsb@cfa.harvard.edu)

import numpy as np
import os
import sys
import tables


############ 
#DATABLOCKS#
############
#descriptions of all datablocks -> add new datablocks here!
#TAG:[HDF5_NAME,DIM]
datablocks = {"POS ":["Coordinates",3], 
	      "VEL ":["Velocities",3],
	      "ID  ":["ParticleIDs",1],
	      "MASS":["Masses",1]}


#####################################################################################################################
#                                                    READING ROUTINES			                            #
#####################################################################################################################





########################### 
#CLASS FOR SNAPSHOT HEADER#
###########################  


class snapshot_header:
    def __init__(self, *args, **kwargs):
        if (len(args) == 1):
            filename = args[0]
            if os.path.exists(filename):
                curfilename=filename
            elif os.path.exists(filename+".hdf5"):
                curfilename = filename+".hdf5"
            elif os.path.exists(filename+".0.hdf5"): 
                curfilename = filename+".0.hdf5"
            else:	
                print("[error] file not found : "+filename)
                sys.exit()
    
            f=tables.open_file(curfilename)
            self.npart = f.root.Header._v_attrs.NumPart_ThisFile 
            self.nall = f.root.Header._v_attrs.NumPart_Total
            self.nall_highword = f.root.Header._v_attrs.NumPart_Total_HighWord
            self.massarr = f.root.Header._v_attrs.MassTable 
            self.time = f.root.Header._v_attrs.Time 
            self.redshift = f.root.Header._v_attrs.Redshift 
            self.boxsize = f.root.Header._v_attrs.BoxSize
            self.filenum = f.root.Header._v_attrs.NumFilesPerSnapshot
            self.omega0 = f.root.Header._v_attrs.Omega0
            self.omegaL = f.root.Header._v_attrs.OmegaLambda
            self.hubble = f.root.Header._v_attrs.HubbleParam
            self.sfr = f.root.Header._v_attrs.Flag_Sfr 
            self.cooling = f.root.Header._v_attrs.Flag_Cooling
            self.stellar_age = f.root.Header._v_attrs.Flag_StellarAge
            self.metals = f.root.Header._v_attrs.Flag_Metals
            self.feedback = f.root.Header._v_attrs.Flag_Feedback
            f.close()

        else:
            #read arguments
            self.npart = kwargs.get("npart")
            self.nall = kwargs.get("nall")
            self.nall_highword = kwargs.get("nall_highword")
            self.massarr = kwargs.get("massarr")
            self.time = kwargs.get("time")
            self.redshift = kwargs.get("redshift")
            self.boxsize = kwargs.get("boxsize")
            self.filenum = kwargs.get("filenum")
            self.omega0 = kwargs.get("omega0")
            self.omegaL = kwargs.get("omegaL")
            self.hubble = kwargs.get("hubble")
            self.sfr = kwargs.get("sfr")
            self.cooling = kwargs.get("cooling")
            self.stellar_age = kwargs.get("stellar_age")
            self.metals = kwargs.get("metals")
            self.feedback = kwargs.get("feedback")
            self.double = kwargs.get("double")
            
            #set default values
            if (self.npart == None):
                self.npart = np.array([0,0,0,0,0,0], dtype="int32")
            if (self.nall == None):				
                self.nall  = np.array([0,0,0,0,0,0], dtype="uint32")
            if (self.nall_highword == None):				
                self.nall_highword = np.array([0,0,0,0,0,0], dtype="uint32")
            if (self.massarr == None):
                self.massarr = np.array([0,0,0,0,0,0], dtype="float64")
            if (self.time == None):				
                self.time = np.array([0], dtype="float64")
            if (self.redshift == None):				
                self.redshift = np.array([0], dtype="float64")
            if (self.boxsize == None):				
                self.boxsize = np.array([0], dtype="float64")
            if (self.filenum == None):
                self.filenum = np.array([1], dtype="int32")
            if (self.omega0 == None):
                self.omega0 = np.array([0], dtype="float64")
            if (self.omegaL == None):
                self.omegaL = np.array([0], dtype="float64")
            if (self.hubble == None):
                self.hubble = np.array([0], dtype="float64")
            if (self.sfr == None):	
                self.sfr = np.array([0], dtype="int32")            
            if (self.cooling == None):	
                self.cooling = np.array([0], dtype="int32")
            if (self.stellar_age == None):	
                self.stellar_age = np.array([0], dtype="int32")
            if (self.metals == None):	
                self.metals = np.array([0], dtype="int32")
            if (self.feedback == None):	
                self.feedback = np.array([0], dtype="int32")
            if (self.double == None):
                self.double = np.array([0], dtype="int32")




##############################
#READ ROUTINE FOR SINGLE FILE#
############################## 
def read_block_single_file(filename, block_name, dim2, parttype=-1, no_mass_replicate=False, fill_block_name="", verbose=False):

    ret_val='ret_val referenced before assignment'

    if (verbose):
        print("[single] reading file           : "+filename)  	
        print("[single] reading                : "+block_name)
      
    head = snapshot_header(filename)
    npart = head.npart
    massarr = head.massarr
    nall = head.nall
    filenum = head.filenum
    doubleflag = 0          
    del head
    
    
    data_slice = slice(None, None, 1)
    
    if (verbose):
        print("[single] data slice: ",end='')
        print(data_slice)
    
 
    f=tables.open_file(filename)
    
    
    #read specific particle type (parttype>=0, non-default)
    if parttype>=0:
        if (verbose):
            print("[single] parttype               : "+parttype) 
        if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0)):
            if (verbose):
                print("[single] replicate mass block")
            ret_val=np.repeat(massarr[parttype], npart[parttype])[data_slice]
        else:		
            part_name='PartType'+str(parttype)
            ret_val = f.root._f_get_child(part_name)._f_get_child(block_name)[data_slice]
        if (verbose):
            print("[single] read particles (total) : ",end='')
            print(ret_val.shape[0]/dim2)
    
    #read all particle types (parttype=-1, default)
    if parttype==-1:
        first=True
        dim1=0
        for parttype in range(0,5):
            part_name='PartType'+str(parttype)
            if (f.root.__contains__(part_name)):
                if (verbose):
                    print("[single] parttype               : ",end='')
                    print(parttype)
                    print("[single] massarr                : ",end='')
                    print(massar)
                    print("[single] npart                  : ",end='')
                    print(npart)


                #replicate mass block per default (unless no_mass_replicate is set)
                if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0) & (no_mass_replicate==False)):
                    if (verbose):
                        print("[single] replicate mass block")
                    if (first):
                        data=np.repeat(massarr[parttype], npart[parttype])
                        dim1+=data.shape[0]
                        ret_val=data
                        first=False
                    else:
                        data=np.repeat(massarr[parttype], npart[parttype])
                        dim1+=data.shape[0]
                        ret_val=np.append(ret_val, data)
                    if (verbose):
                        print("[single] read particles (total) : ",end='')
                        print(ret_val.shape[0]/dim2)
                    if (doubleflag==0):
                        ret_val=ret_val.astype("float32")


                    #fill fill_block_name with zeros if fill_block_name is set and particle type is present and fill_block_name not already stored in file for that particle type
                if ((block_name==fill_block_name) & (block_name!="Masses") & (npart[parttype]>0) & (f.root._f_get_child(part_name).__contains__(block_name)==False)):
                    if (verbose):
                        print("[single] replicate block name",end='')
                        print(fill_block_name)
                    if (first):
                        data=np.repeat(0.0, npart[parttype]*dim2)
                        dim1+=data.shape[0]
                        ret_val=data
                        first=False
                    else:
                        data=np.repeat(0.0, npart[parttype]*dim2)
                        dim1+=data.shape[0]
                        ret_val=np.append(ret_val, data)
                    if (verbose):
                        print("[single] read particles (total) : ",end='')
                        print(ret_val.shape[0]/dim2)
                    if (doubleflag==0):
                        ret_val=ret_val.astype("float32")
    
                #default: just read the block
                if (f.root._f_get_child(part_name).__contains__(block_name)):
                    if (first):
                        data=f.root._f_get_child(part_name)._f_get_child(block_name)[:]
                        dim1+=data.shape[0]
                        ret_val=data
                        first=False
                    else:
                        data=f.root._f_get_child(part_name)._f_get_child(block_name)[:]
                        dim1+=data.shape[0]
                        ret_val=np.append(ret_val, data)
                    if (verbose):
                        print("[single] read particles (total) : ",end='')
                        print(ret_val.shape[0]/dim2)
    
        if ((dim1>0) & (dim2>1)):
            ret_val=ret_val.reshape(dim1,dim2)

    f.close()

    if type(ret_val)==str:
        print(ret_val)
    return ret_val

##############
#READ ROUTINE#
##############
def read_block(filename, block, parttype=-1, no_mass_replicate=False, fill_block="", verbose=False):
    if (verbose):
        print("reading block          : ",end='')
        print(block)
    
    if parttype not in [-1,0,1,2,3,4,5]:
        print("[error] wrong parttype given")
        sys.exit()
    
    curfilename=filename+".hdf5"
    
    if os.path.exists(curfilename):
        multiple_files=False
    elif os.path.exists(filename+".0"+".hdf5"):
        curfilename = filename+".0"+".hdf5"
        multiple_files=True
    else:
        print("[error] file not found : "+filename)
        sys.exit()
    
    head = snapshot_header(curfilename)
    filenum = head.filenum
    del head
    
    
    if (block in datablocks.keys()):
        block_name=datablocks[block][0]
        dim2=datablocks[block][1]
        first=True
        if (verbose):
            print("Reading HDF5           : ",end='')
            print(block_name)
            print("Data dimension         : ",end='')
            print(dim2)
            print("Multiple file          : ",end='')
            print(multiple_files)
    else:
        print("[error] Block type ",end='')
        print(block,end='')
        print( "not known!")
        sys.exit()
    
    fill_block_name=""
    if (fill_block!=""):
        if (fill_block in datablocks.keys()):
            fill_block_name=datablocks[fill_block][0]
            dim2=datablocks[fill_block][1]
            if (verbose):
                print("Block filling active   : ",end='')
                print(fill_block_name)
    
    
    if (multiple_files):	
        first=True
        dim1=0
        for num in range(0,filenum):
            curfilename=filename+"."+str(num)+".hdf5"
            if (verbose):
                print("Reading file           : ",end='')
                print(num,end='')
                print(curfilename)
            if (first):
                data = read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)
                dim1+=data.shape[0]
                ret_val = data
                first = False 
            else:	 
                data = read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)
                dim1+=data.shape[0]
                ret_val=np.append(ret_val, data)
            if (verbose):
                print("Read particles (total) : ",end='')
                print(ret_val.shape[0]/dim2)
        
        if ((dim1>0) & (dim2>1)):
            ret_val=ret_val.reshape(dim1,dim2)	
    else:
        ret_val=read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)
    
    return ret_val

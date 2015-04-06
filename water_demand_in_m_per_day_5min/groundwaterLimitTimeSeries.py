#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import calendar
import time
import re
import subprocess
import pcraster as pcr
import netCDF4 as nc
import numpy as np
import virtualOS as vos

class ConvertMapsToNetCDF4():
    
    def __init__(self,cloneMapFile,attribute=None,cellSizeInArcMinutes=None):
        		
        # cloneMap
        cloneMap = pcr.readmap(cloneMapFile)
        cloneMap = pcr.boolean(1.0)
        
        # properties of the clone map
        # - number of rows and columns
        self.nrRows       = np.round(pcr.clone().nrRows())    
        self.nrCols       = np.round(pcr.clone().nrCols())  
        # - upper right coordinate, unit: arc degree ; must be integer (without decimals)
        self.minLongitude = np.round(pcr.clone().west() , 0)         
        self.maxLatitude  = np.round(pcr.clone().north(), 0)
        # - cell resolution, unit: arc degree
        self.cellSize     = pcr.clone().cellSize()
        if cellSizeInArcMinutes != None: self.cellSize = cellSizeInArcMinutes / 60.0 
        # - lower right coordinate, unit: arc degree ; must be integer (without decimals)
        self.maxLongitude = np.round(self.minLongitude + self.cellSize*self.nrCols, 0)         
        self.minLatitude  = np.round(self.maxLatitude  - self.cellSize*self.nrRows, 0)
        
        # latitudes and longitudes for netcdf files
        latMin = self.minLatitude  + self.cellSize / 2
        latMax = self.maxLatitude  - self.cellSize / 2
        lonMin = self.minLongitude + self.cellSize / 2
        lonMax = self.maxLongitude - self.cellSize / 2
        self.longitudes = np.arange(lonMin,lonMax+self.cellSize, self.cellSize)
        self.latitudes=   np.arange(latMax,latMin-self.cellSize,-self.cellSize)
        
        # netCDF format and attributes:
        self.format = 'NETCDF4'
        self.attributeDictionary = {}
        if attribute == None:
            self.attributeDictionary['institution'] = "None"
            self.attributeDictionary['title'      ] = "None"
            self.attributeDictionary['description'] = "None"
        else:
            self.attributeDictionary = attribute

    def createNetCDF(self,ncFileName,varNames,varUnits):

        rootgrp= nc.Dataset(ncFileName,'w',format= self.format)

        #-create dimensions - time is unlimited, others are fixed
        rootgrp.createDimension('time',None)
        rootgrp.createDimension('lat',len(self.latitudes))
        rootgrp.createDimension('lon',len(self.longitudes))

        date_time= rootgrp.createVariable('time','f4',('time',))
        date_time.standard_name= 'time'
        date_time.long_name= 'Days since 1901-01-01'

        date_time.units= 'Days since 1901-01-01' 
        date_time.calendar= 'standard'

        lat= rootgrp.createVariable('lat','f4',('lat',))
        lat.long_name= 'latitude'
        lat.units= 'degrees_north'
        lat.standard_name = 'latitude'

        lon= rootgrp.createVariable('lon','f4',('lon',))
        lon.standard_name= 'longitude'
        lon.long_name= 'longitude'
        lon.units= 'degrees_east'

        lat[:]= self.latitudes
        lon[:]= self.longitudes

        for iVar in range(0,len(varNames)):      
            shortVarName = varNames[iVar]
            var= rootgrp.createVariable(shortVarName,'f4',('time','lat','lon',) ,fill_value=vos.MV,zlib=False)
            var.standard_name = shortVarName
            var.long_name = shortVarName
            var.units = varUnits[iVar]

        attributeDictionary = self.attributeDictionary
        for k, v in attributeDictionary.items():
          setattr(rootgrp,k,v)

        rootgrp.sync()
        rootgrp.close()

    def writePCR2NetCDF(self,ncFileName,varName,varField,timeStamp,posCnt):

        #-write data to netCDF
        rootgrp= nc.Dataset(ncFileName,'a')    

        shortVarName= varName        

        date_time= rootgrp.variables['time']
        date_time[posCnt]= nc.date2num(timeStamp,date_time.units,date_time.calendar)

        rootgrp.variables[shortVarName][posCnt,:,:]= (varField)

        rootgrp.sync()
        rootgrp.close()

if __name__ == "__main__":
    
    # clone, landmask and cell area files
    landmaskFile     = "/data/hydroworld/PCRGLOBWB20/input30min/routing/lddsound_30min.map"
    cloneMapFileName = landmaskFile 
    cellAreaFile     = "/data/hydroworld/PCRGLOBWB20/input30min/routing//cellarea30min.map"
    
    # attribute for netCDF files 
    attributeDictionary = {}
    attributeDictionary['title'      ]  = "Yearly annual 30 arc min pumping capacity/limit."
    attributeDictionary['institution']  = "Dept. of Physical Geography, Utrecht University"
    attributeDictionary['source'     ]  = "None"
    attributeDictionary['history'    ]  = "None"
    attributeDictionary['references' ]  = "(Wada et al., 2010; Wada et al., 2012)"
    attributeDictionary['comment'    ]  = "This file was converted and resampled from the files provided by Yoshi Wada in October 2014."
    # additional attribute defined in PCR-GLOBWB 
    attributeDictionary['description']  = "None"

    # input files
    inputDirectory = "/home/sutan101/data/data_from_yoshi/groundwater/"
    inputFiles     = 'waterdemand_30min_groundwaterabstraction_yearly.nc'
    inputVarNames  = 'gwab'
    #
    # Note that the unit in input files are in mcm/month, for livestock, gross water demand is equal to netto water demand (everything is consumed)

    # output files
    ncFileName = 'annual_groundwater_abstraction_limit_30min.nc'
    varNames = ['regional_pumping_limit','regional_pumping_limit_masked','region_ids','region_ids_masked']
    varUnits = ['bcm.year-1','bcm.year-1',"30minID","30minID"]            

    # start year and end year
    staYear = 1960
    endYear = 2001

    # output and temporary directories
    out_directory = "/scratch/edwin/data/data_from_yoshi/groundwater/groundwater_abstraction_limit_30min/"
    tmp_directory = out_directory+"/tmp/"
    
    # netcdf file name, including its directory
    ncFileName = out_directory+"/"+ncFileName 

    # prepare output and temporary directories:
    try:
        os.makedirs(out_directory)    
    except:
        pass
    try:
        os.makedirs(tmp_directory)    
    except:
        pass
            
    # initiate the netcd file and object: 
    tssNetCDF = ConvertMapsToNetCDF4(cloneMapFile = cloneMapFileName, attribute = attributeDictionary)
    tssNetCDF.createNetCDF(ncFileName,varNames,varUnits)

    index = 0 # for posCnt
    
    # set clone and define land mask region
    pcr.setclone(landmaskFile)
    landmask = pcr.defined(pcr.readmap(landmaskFile))
    landmask = pcr.ifthen(landmask, landmask)
    
    # cell area (m2)
    cellArea = vos.readPCRmapClone(cellAreaFile,
                                   cloneMapFileName,tmp_directory)
    cellArea = pcr.ifthen(landmask,cellArea)
    
    for iYear in range(staYear,endYear+1):

        # time stamp
        timeStamp = datetime.datetime(int(iYear),int(1),int(1),int(0))
        fulldate = '%4i-%02i-%02i' %(int(iYear),int(1),int(1))
        print fulldate

        # index for time object in the netcdf file:
        index = index + 1

        # reading values from the input netcdf files (30min)
        abstraction_volume_30min = pcr.roundup(
                                   vos.netcdf2PCRobjClone(inputDirectory+inputFiles,\
                                                          inputVarNames,
                                                          fulldate,
                                                          None,
                                                          cloneMapFileName)) * 0.001   # unit: bcm/year
        
        abstraction = pcr.cover(abstraction_volume_30min, 0.0)
        
        # use window maximum to be in the conservative side:
        window_size = 1.0
        abstraction = pcr.windowmaximum(abstraction, 1.0) 
        
        # covering the map with zero
        pcrValue = pcr.cover(abstraction, 0.0)  # unit: m/day                       

        # the value should be higher than the previous yeat value
        if iYear > staYear:
            pcrValue = pcr.max(preValue, pcrValue)
            print iYear 
        else:
            preValue = pcrValue
            print 'first year'
        
        region_ids = pcr.uniqueid(pcr.boolean(1.0))
        region_ids_masked = pcr.ifthen(landmask, region_ids)
        
        regional_pumping_limit = pcrValue
        regional_pumping_limit_masked = pcr.ifthen(landmask, regional_pumping_limit)

        for iVar in range(0,len(varNames)):      
            var = varNames[iVar]
            pcrRead = vars()[str(var)]
            varField = pcr.pcr2numpy(pcrRead, vos.MV)
            tssNetCDF.writePCR2NetCDF(ncFileName,var,varField,timeStamp,posCnt = index - 1)

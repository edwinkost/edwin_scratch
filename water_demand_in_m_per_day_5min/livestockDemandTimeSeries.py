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
        # - the cloneMap must be at 5 arc min resolution
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
    landmask05minFile    = "/data/hydroworld/PCRGLOBWB20/input5min/routing/lddsound_05min.map"
    cloneMapFileName     = landmask05minFile 
    cellSizeInArcMinutes = 5.0 
    cellArea05minFile    = "/data/hydroworld/PCRGLOBWB20/input5min/routing/cellsize05min.correct.map"
    
    # unique ids for every 30 arc minute grid (provided in scalar values):
    uniqueIDs30minFile = "/data/hydroworld/others/irrigationZones/half_arc_degree/uniqueIds30min.map "
    
    # attribute for netCDF files 
    attributeDictionary = {}
    attributeDictionary['title'      ]  = "Monthly livestock water demand"
    attributeDictionary['institution']  = "Dept. of Physical Geography, Utrecht University"
    attributeDictionary['source'     ]  = "None"
    attributeDictionary['history'    ]  = "None"
    attributeDictionary['references' ]  = "None"
    attributeDictionary['comment'    ]  = "This file has 5 arc minute resolution and was prepared by Edwin H. Sutanudjaja in April 2015." 
    attributeDictionary['comment'    ] += " It is converted and resampled from 30 arc-minutes water demand files provided by Yoshi Wada in October 2014."
    # additional attribute defined in PCR-GLOBWB 
    attributeDictionary['description'] = "None"

    # input files
    inputDirectory = "/data/hydroworld/basedata/human/water_demand_wada_et_al_2014/"
    inputFiles     = ['pcrglobwb_historical_PLivUse_monthly_1960_2010.nc4','pcrglobwb_historical_PLivUse_monthly_1960_2010.nc4']
    inputVarNames  = ['PLivWN','PLivWN']
    #
    # Note that the unit in input files are in mcm/month, for livestock, gross water demand is equal to netto water demand (everything is consumed)

    # output files
    ncFileName = 'livestock_water_demand_version_april_2015.nc'
    varNames   = ['livestockNettoDemand','livestockGrossDemand']
    varUnits   = ['m.day-1','m.day-1']            

    # start year and end year
    staYear = 1960
    endYear = 2010

    # output and temporary directories
    out_directory = "/home/sutan101/data/data_from_yoshi/water_demand/water_demand_in_m_per_day_05min/"
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
    tssNetCDF = ConvertMapsToNetCDF4(cloneMapFile = cloneMapFileName, attribute = attributeDictionary, cellSizeInArcMinutes = cellSizeInArcMinutes)
    tssNetCDF.createNetCDF(ncFileName,varNames,varUnits)

    index = 0 # for posCnt
    
    # set clone and define land mask region
    pcr.setclone(landmask05minFile)
    landmask = pcr.defined(pcr.readmap(landmask05minFile))
    landmask = pcr.ifthen(landmask, landmask)
    
    # cell area at 5 arc min resolution
    cellArea = vos.readPCRmapClone(cellArea05minFile,
                                   cloneMapFileName,tmp_directory)
    cellArea = pcr.ifthen(landmask,cellArea)
    
    # ids for every 30 arc min grid:
    uniqueIDs30min = vos.readPCRmapClone(uniqueIDs30minFile,
                                         cloneMapFileName,tmp_directory) 
    uniqueIDs30min = pcr.nominal(pcr.ifthen(landmask, uniqueIDs30min))
    
    for iYear in range(staYear,endYear+1):
        for iMonth in range(1,12+1):
            timeStamp = datetime.datetime(int(iYear),int(iMonth),int(1),int(0))

            fulldate = '%4i-%02i-%02i' %(int(iYear),int(iMonth),int(1))
            print fulldate

            monthRange = float(calendar.monthrange(int(iYear), int(iMonth))[1])
            print(monthRange)
            
            index = index + 1
            for iVar in range(0,len(varNames)):      
                
                # reading values from the input netcdf files (30min)
                demand_volume_30min = vos.netcdf2PCRobjClone(inputDirectory+inputFiles[iVar],\
                                                             inputVarNames[iVar],
                                                             fulldate,
                                                             None,
                                                             cloneMapFileName) * 1000.*1000./ monthRange   # unit: m3/day
                demand_volume_30min = pcr.ifthen(landmask, demand_volume_30min)
                
                # demand in m/day
                demand = demand_volume_30min /\
                         pcr.areatotal(cellArea, uniqueIDs30min)
                
                # covering the map with zero
                pcrValue = pcr.cover(demand, 0.0)  # unit: m/day                       

                # convert values to pcraster object
                varField = pcr.pcr2numpy(pcrValue, vos.MV)

                # write values to netcdf files
                tssNetCDF.writePCR2NetCDF(ncFileName,varNames[iVar],varField,timeStamp,posCnt = index - 1)

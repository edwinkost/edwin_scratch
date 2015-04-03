#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import calendar
import time
import re
import subprocess
import netCDF4 as nc
import numpy as np
import pcraster as pcr
import virtualOS as vos

class ConvertMapsToNetCDF4():
    
    def __init__(self,cloneMapFile,attribute=None,cellSize=None):
        		
        # cloneMap
        cloneMap = pcr.readmap(cloneMapFile)
        cloneMap = pcr.boolean(1.0)
        
        # latitudes and longitudes
        self.latitudes  = np.unique(pcr.pcr2numpy(pcr.ycoordinate(cloneMap), vos.MV))[::-1]
        self.longitudes = np.unique(pcr.pcr2numpy(pcr.xcoordinate(cloneMap), vos.MV))
        
        # 

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
        #~ for iVar in range(1,1+1):      
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
    
    cloneMapUsed = '/data/hydroworld/PCRGLOBWB20/input30min/routing/lddsound_30min.map'
    cellAreaFile = '/data/hydroworld/PCRGLOBWB20/input30min/routing/cellarea30min.map'
    
    # attribute for netCDF files 
    attributeDictionary = {}
    attributeDictionary['title'      ] = "Monthly domestic water demand"
    attributeDictionary['institution'] = "Dept. of Physical Geography, Utrecht University"
    attributeDictionary['source'     ] = "None"
    attributeDictionary['history'    ] = "None"
    attributeDictionary['references' ] = "None"
    attributeDictionary['comment'    ] = "Converted from netcdf files provided by Yoshi Wada in October 2014."
    # additional attribute defined in PCR-GLOBWB 
    attributeDictionary['description'] = "None"

    # input files
    inputDirectory = "/data/hydroworld/basedata/human/water_demand_wada_et_al_2014/"
    inputFiles     = ['pcrglobwb_historical_PDomUse_monthly_1960_2010.nc4','pcrglobwb_historical_PDomWW_monthly_1960_2010.nc4']
    inputVarNames  = ['PDomUse','PDomWW']
    #
    # Note that the unit in input files are in mcm/month

    # output files
    ncFileName = 'domestic_water_demand_version_october_2014.nc'
    varNames   = ['domesticNettoDemand','domesticGrossDemand']

    varUnits = ['m.day-1',
                'm.day-1']            

    staYear = 1960
    endYear = 2010
    
    tssNetCDF = ConvertMapsToNetCDF4(cloneMapFile = cloneMapUsed, attribute = attributeDictionary)
    tssNetCDF.createNetCDF(ncFileName,varNames,varUnits)

    index = 0 # for posCnt
    
    pcr.setclone(cloneMapUsed)
    cellArea05min = pcr.readmap(cellAreaFile)

    
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
                pcrValue = vos.netcdf2PCRobjClone(inputDirectory+inputFiles[iVar],\
                                                  inputVarNames[iVar],
                                                  fulldate) * 1000.*1000./ monthRange   # unit: m3/day
                pcrValue = pcr.cover(pcrValue / cellArea30min, 0.0)                     # unit: m1/day                       

                # use the maximum value per every 30 arcmin grid
                pcrValue = pcr.areamximum(pcrValue, uniqueIds30min)





                # convert values to pcraster object
                varField = pcr.pcr2numpy(pcrValue, vos.MV)

                # write values to netcdf files
                tssNetCDF.writePCR2NetCDF(ncFileName,varNames[iVar],varField,timeStamp,posCnt = index - 1)

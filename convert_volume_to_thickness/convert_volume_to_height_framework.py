#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import calendar
import numpy as np

import pcraster as pcr
from pcraster.framework import DynamicModel

from outputNetcdf import OutputNetcdf
import virtualOS as vos

class ConvertVolumeToHeightFramework(DynamicModel):

    def __init__(self, input_netcdf,\
                       output_netcdf,\
                       modelTime,\
                       tmpDir = "/dev/shm/"):
        DynamicModel.__init__(self) 

        self.input_netcdf = input_netcdf
        self.output_netcdf = output_netcdf 
        self.tmpDir = tmpDir
        self.modelTime = modelTime

        # set clone
        self.clone_map_file = self.input_netcdf['cell_area']
        pcr.setclone(self.clone_map_file)
        self.clone = {}
        self.clone['cellsize'] = pcr.clone().cellSize() ; print self.clone['cellsize']
        self.clone['rows']     = int(pcr.clone().nrRows()) 
        self.clone['cols']     = int(pcr.clone().nrCols())
        self.clone['xUL']      = round(pcr.clone().west(), 2)
        self.clone['yUL']      = round(pcr.clone().north(), 2)

        # cell area (unit: m2)
        self.cell_area = pcr.readmap(self.input_netcdf['cell_area'])

        # an object for netcdf reporting
        self.output = OutputNetcdf(self.clone, self.output_netcdf)       
        
        # preparing the netcdf file and make variable:
        self.output.createNetCDF(self.output_netcdf['file_name'],
                                 self.output_netcdf['gross_variable_name'],
                                 self.output_netcdf['variable_unit'])
        self.output.addNewVariable(self.output_netcdf['file_name'],
                                   self.output_netcdf['netto_variable_name'],
                                   self.output_netcdf['variable_unit'])
        
    def initial(self): 
        pass

    def dynamic(self):
        
        # update model time using the current pcraster timestep value
        self.modelTime.update(self.currentTimeStep())

        # reading gross and netto values:
        if self.modelTime.isLastDayOfMonth():

            gross_value = vos.netcdf2PCRobjClone(ncFile  = self.input_netcdf['gross_file_name'],
                                                 varName = self.input_netcdf['gross_variable_name'],
                                                 dateInput = str(self.modelTime.fulldate))

            netto_value = vos.netcdf2PCRobjClone(ncFile  = self.input_netcdf['netto_file_name'],
                                                 varName = self.input_netcdf['netto_variable_name'],
                                                 dateInput = str(self.modelTime.fulldate))
            
            # covering with zero and convert the unit to 
            gross_value = pcr.cover(gross_value, 0.0)/self.cell_area                                     
            netto_value = pcr.cover(netto_value, 0.0)/self.cell_area                                     

        # reporting
        if self.modelTime.isLastDayOfMonth():

            # time stamp 
            timestepPCR = self.modelTime.timeStepPCR
            timeStamp = datetime.datetime(self.modelTime.year,\
                                          self.modelTime.month,\
                                          self.modelTime.day,0)
            # to netcdf 
            self.output.data2NetCDF(self.output_netcdf['file_name'],\
                                    self.output_netcdf['gross_variable_name'],\
                                    pcr.pcr2numpy(gross_value, vos.MV),\
                                    timeStamp, self.modelTime.timeStepPCR -1)
            # write netto value to netcdf 
            self.output.data2NetCDF(self.output_netcdf['file_name'],\
                                    self.output_netcdf['netto_variable_name'],\
                                    pcr.pcr2numpy(netto_value, vos.MV),\
                                    timeStamp, self.modelTime.timeStepPCR -1)

        # closing the file at the end of
        if self.modelTime.isLastTimeStep(): self.output.close(self.output_netcdf['file_name'])


#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

# pcraster dynamic framework is used:
from pcraster.framework import DynamicFramework

# classes used in this script
from convert_volume_to_height_framework import ConvertVolumeToHeightFramework

# time object
from currTimeStep import ModelTime

# utility module:
import virtualOS as vos

# starting and end dates
startDate = "1960-01-01" # "2008-01-01" # "1990-01-01" #YYYY-MM-DD
endDate   = "2010-12-31" # "2013-12-31" # "2014-10-31" #YYYY-MM-DD


# input netcdf files of water demand data:
input_netcdf = {}
input_netcdf['gross_file_name']     = "/scratch/edwin/GLOWASIS_water_demand/06min/Global-Water-Demand-Domestic-6min-Globe-domw_month.nc4"
input_netcdf['gross_variable_name'] = "domw"
input_netcdf['netto_file_name']     = "/scratch/edwin/GLOWASIS_water_demand/06min/Global-Water-Demand-Domestic-6min-Globe-domn_month.nc4"
input_netcdf['netto_variable_name'] = "domn"

# cell area in m2 and cell resolution in arc_degree 
input_netcdf['cell_area']        = "/scratch/edwin/GLOWASIS_water_demand/06min/clone_and_cell_area_6min/Global_CellArea-m2_06min.map"
input_netcdf['cell_resolution']  = 6./60.

# output netcdf files:
output_netcdf = {}
output_netcdf['format']    = "NETCDF4"
output_netcdf['zlib']      = True
output_netcdf['folder']              = "/scratch/edwin/GLOWASIS_water_demand/06min_m_per_day/domestic/"
output_netcdf['file_name']           = output_netcdf['folder']+"/"+"domestic"+"_water_demand_6min_meter_per_day.nc"
output_netcdf['gross_variable_name'] = 'domesticGrossDemand'
output_netcdf['netto_variable_name'] = 'domesticNettoDemand'
output_netcdf['variable_unit']    = 'm.day-1'
output_netcdf['netcdf_attribute'] = {}
output_netcdf['netcdf_attribute']['title'      ]  = "Monthly domestic water demand."
output_netcdf['netcdf_attribute']['institution']  = "Department of Physical Geography, Utrecht University"
output_netcdf['netcdf_attribute']['source'     ]  = "GLOWASIS water demand."
output_netcdf['netcdf_attribute']['history'    ]  = "The data were provided by Yoshihide Wada (Y.Wada@uu.nl) in cubic meter per day and then converted by Edwin H. Sutanudjaja (E.H.Sutanudjaja@uu.nl) to meter per day."
output_netcdf['netcdf_attribute']['references' ]  = "Wada, Y., Wisser, D., and Bierkens, M. F. P.: Global modeling of withdrawal, allocation and "
output_netcdf['netcdf_attribute']['references' ] += "consumptive use of surface water and groundwater resources, Earth Syst. Dynam., 5, 15-40, doi:10.5194/esd-5-15-2014, 2014."
output_netcdf['netcdf_attribute']['comment'    ]  = "GLOWASIS water demand."
output_netcdf['netcdf_attribute']['comment'    ] += "For further usage, please contact Yoshihide Wada (Y.Wada@uu.nl). "
output_netcdf['netcdf_attribute']['description'] =  "GLOWASIS water demand."

# make an output folder
cleanOutputFolder = False
try:
    os.makedirs(output_netcdf['folder'])
except:
    if cleanOutputFolder: os.system('rm -r '+str(output_netcdf['folder'])+"/*")

# make a temporary folder 
tmpDir = output_netcdf['folder']+"/"+"tmp/"
try:
    os.makedirs(tmpDir)
except:
    os.system('rm -r '+str(output_netcdf['folder'])+"/tmp/*")

def main():
    
    # time object
    modelTime = ModelTime() # timeStep info: year, month, day, doy, hour, etc
    modelTime.getStartEndTimeSteps(startDate, endDate)
    
    # converting model
    convertModel = ConvertVolumeToHeightFramework(input_netcdf, \
                                                  output_netcdf, \
                                                  modelTime, \
                                                  tmpDir)
    dynamic_framework = DynamicFramework(convertModel, modelTime.nrOfTimeSteps)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()

if __name__ == '__main__':
    sys.exit(main())

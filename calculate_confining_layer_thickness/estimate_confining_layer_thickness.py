
import pcraster as pcr
import netCDF4 as nc

# clone map
clone_map_file_name = "/projects/0/dfguu/data/hydroworld/PCRGLOBWB20/input5min/routing/lddsound_05min.map"
pcr.setclone(clone_map_file_name)

# aquifer thickness map:
aquifer_thickness_file_name = "/projects/0/dfguu/users/edwin/data/aquifer_properties/thickness_05min.map"
aquifer_thickness = pcr.readmap(aquifer_thickness_file_name)

# extent of confining layer
extent_of_confining_layer_file_name = "/home/edwin/data/inge_confining_layer_parameters/conflayers4.map"
confining_layer_extent = pcr.boolean(pcr.readmap(extent_of_confining_layer_file_name))

# thickness of confining layer = 10 percent from the first 250 m
confining_layer_thickness = pcr.ifthen(confining_layer_extent, pcr.min(250.0, aquifer_thickness)) * 0.10

# extrapolate
confining_layer_thickness = pcr.cover(confining_layer_thickness, pcr.windowaverage(pcr.cover(confining_layer_thickness, 0.0), 0.50))
confining_layer_thickness = pcr.cover(confining_layer_thickness, 0.0)
confining_layer_thickness_output_filename = "/home/edwin/data/inge_confining_layer_parameters/confining_layer_thickness_edwin.map"
pcr.report(confining_layer_thickness, confining_layer_thickness_output_filename)

# masking only to the landmask 
landmask_file_name = "/projects/0/dfguu/data/hydroworld/PCRGLOBWB20/input5min/routing/lddsound_05min.map"
landmask = pcr.defined(landmask_file_name)
confining_layer_thickness_masked_output_filename = "/home/edwin/data/inge_confining_layer_parameters/confining_layer_thickness_edwin.masked.map"
pcr.report(pcr.ifthen(landmask, confining_layer_thickness), confining_layer_thickness_masked_output_filename)

globals().clear()
clear all
os.system("clear")

#-------------------  import packages ----------------------------------------------------------------------------------
import sys
import os
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from itertools import islice
from mpl_toolkits.basemap import Basemap
import fiona
import rasterio.mask
import rasterio
import pyproj
from rasterio.transform import Affine
from scipy.interpolate import interp1d

#-----------------------------set file paths----------------------------------------------------------------------------
file_path = '/home/guanl/Desktop/MSP/Climatology'
output_path = '/home/guanl/Desktop/MSP/Climatology/'
grd_file = os.path.join(file_path, 'nep35_reord_latlon_wgeo.ngh')
tri_file = os.path.join(file_path, 'nep35_reord.tri')
#tem_file = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2.dat')
#tem_reformat = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2_reformat')

#------------------------------run functions----------------------------------------------------------------------------
#Specify index
output_folder = 'S_spr'
season = 'spr'


#Read and plot triangle grid
clim_data = read_climatologies(file_path = '/home/guanl/Desktop/MSP/Climatology', output_folder = 'S_spr', season = 'spr')

clim_data_1 = bottom_values(clim_data)

#plot climatology on triangle grid
plot_clim_triangle(clim_data_1, file_path = '/home/guanl/Desktop/MSP/Climatology', left_lon = -160, right_lon = -102, bot_lat = 25, top_lat = 62, output_folder = 'S_spr', season = 'spr', depth = 'bottom')

# Convert and plot climatology on regular grid
left_lon, right_lon, bot_lat, top_lat = [-140, -120, 45, 56]

clim_data_r = triangle_to_regular(clim_data_1, file_path = '/home/guanl/Desktop/MSP/Climatology', left_lon = -140, right_lon = -120, bot_lat = 45, top_lat = 56, output_folder = 'S_spr', season = 'spr', depth = 'bottom')

# Convert to raster layer and save in .tif format
convert_to_tif(clim_data_r, file_path = '/home/guanl/Desktop/MSP/Climatology', output_folder = 'S_spr', season = 'spr', depth = 'bottom')

#use EEZ polygon to clip on GeoTiff
EEZ_clip(file_path = '/home/guanl/Desktop/MSP/Climatology', output_folder = 'S_spr', season = 'spr', depth = 'bottom')



#------------------ Read and plot climatology on triangle grid-----------------------------------------------------------
def read_climatologies(file_path, output_folder, season):
    grid_filename = os.path.join(file_path, 'nep35_reord_latlon_wgeo.ngh')
    tri_filename = os.path.join(file_path, 'nep35_reord.tri')

    data = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                          names=['node', 'lon', 'lat', 'type', 'depth',
                                's1', 's2', 's3', 's4', 's5', 's6'],
                          delimiter="", skip_header=3)

    tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3))-1 #python starts from 0
    array_filename = os.path.join(file_path, output_folder + '/nep35_sal_' + season + '_extrap2_reformat.npy')

    array = np.load(array_filename)
    array_t = array[1:]
    array_t = np.transpose(array_t)
    grid_depth = abs(array[0])
    array_t = np.vstack((array_t, data['depth']))
 #   for i in range(0, 51, 1):
 #       array_t[i] = np.where( array_t[52] < grid_depth[i], np.nan, array_t[i]) #replace the value below bottom depth with nan

    # create a data dictionary, and write data into dictionary
    data_dict = dict()
    data_dict['node_number'] = data['node'] - 1 # use node_number as Key
    data_dict['depth_in_m'] = data['depth']
    data_dict['y_lat'] = data['lat']
    data_dict['x_lon'] = data['lon']
    data_dict['grid_depth'] = abs(array[0])

    #write index for each grid depth
    for i in range(0, 52, 1):
        variable_name = 'grid_depth_' + str(int(abs(grid_depth[i]))) + 'm'
        data_dict[variable_name] = array_t[i]

    tri = mtri.Triangulation(data_dict['x_lon'], data_dict['y_lat'], tri_data) # attributes: .mask, .triangles, .edges, .neighbors
    #min_circle_ratio = 0.1
    #mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    #tri.set_mask(mask)
    data_dict['triangles'] = tri.triangles
    plt.triplot(tri, color='0.7', lw = 0.2)  #check grid plot
    plt.show()

    return data_dict

#---------------------------------------- calculate bottom values--------------------------------------------------------

def bottom_values(data_dict):
    n = data_dict['depth_in_m'].shape[0]
    var = np.zeros(n)
    i = 0
    for i in range(0, n, 1):
        bottom_depth = data_dict['depth_in_m'][i]
        if 5 < bottom_depth < 5000:
            n_before = np.where(data_dict['grid_depth'] < bottom_depth)[0][0]
            n_after = n_before + 1
            grid_depth_before = data_dict['grid_depth'][n_before]
            grid_depth_after = data_dict['grid_depth'][n_after]
            var_value_before = data_dict['grid_depth_' + str(int(grid_depth_before)) + 'm'][i]
            var_value_after = data_dict['grid_depth_' + str(int(grid_depth_after)) + 'm'][i]
            depth_interp = np.array([grid_depth_before, grid_depth_after])
            values_interp = np.array([var_value_before, var_value_after])
            l_interp = interp1d(depth_interp, values_interp, kind='linear', fill_value = 'extrapolate')
            if l_interp(bottom_depth) < 36:
                var[i] = l_interp(bottom_depth)
            else:
                n_before = np.where(data_dict['grid_depth'] < bottom_depth)[0][0]
                n_after = np.where(data_dict['grid_depth'] > bottom_depth)[0][-1]
                grid_depth_before = data_dict['grid_depth'][n_before]
                grid_depth_after = data_dict['grid_depth'][n_after]
                var_value_before = data_dict['grid_depth_' + str(int(grid_depth_before)) + 'm'][i]
                var_value_after = data_dict['grid_depth_' + str(int(grid_depth_after)) + 'm'][i]
                depth_interp = np.array([grid_depth_before, grid_depth_after])
                values_interp = np.array([var_value_before, var_value_after])
                l_interp = interp1d(depth_interp, values_interp, kind='linear')
                var[i] = l_interp(bottom_depth)
        elif bottom_depth <= 5:
            #n_before = np.where(data_dict['grid_depth'] < bottom_depth)[0][0]
            #n_after = np.where(data_dict['grid_depth'] > bottom_depth)[0][-1]
            #grid_depth_before = data_dict['grid_depth'][n_before]
            #grid_depth_after = data_dict['grid_depth'][n_after]
            grid_depth_before = data_dict['grid_depth'][51]
            grid_depth_after = data_dict['grid_depth'][50]
            var_value_before = data_dict['grid_depth_' + str(int(grid_depth_before)) + 'm'][i]
            var_value_after = data_dict['grid_depth_' + str(int(grid_depth_after)) + 'm'][i]
            depth_interp = np.array([grid_depth_before, grid_depth_after])
            values_interp = np.array([var_value_before, var_value_after])
            l_interp = interp1d(depth_interp, values_interp, kind='linear', fill_value = 'extrapolate')
            var[i] = l_interp(bottom_depth)
        elif bottom_depth >= 5000:
            grid_depth_before = data_dict['grid_depth'][1]
            grid_depth_after = data_dict['grid_depth'][0]
            var_value_before = data_dict['grid_depth_' + str(int(grid_depth_before)) + 'm'][i]
            var_value_after = data_dict['grid_depth_' + str(int(grid_depth_after)) + 'm'][i]
            depth_interp = np.array([grid_depth_before, grid_depth_after])
            values_interp = np.array([var_value_before, var_value_after])
            l_interp = interp1d(depth_interp, values_interp, kind='linear', fill_value = 'extrapolate')
            var[i] = l_interp(bottom_depth)

    data_dict['grid_depth_bottom'] = var

    return data_dict



#-----------------------------Plot Climatology with unstructured triangle grid------------------------------------------
left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]

def plot_clim_triangle(data_dict, file_path, left_lon, right_lon, bot_lat, top_lat, output_folder, season, depth):
    tri_filename = os.path.join(file_path, 'nep35_reord.tri')
    tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3))-1
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    # lcc: Lambert Conformal Projection;
    # cyl: Equidistant Cylindrical Projection
    # merc: Mercator Projection

    x_lon = data_dict['x_lon']
    y_lat = data_dict['y_lat']
    xpt, ypt = m(x_lon, y_lat) #convert lat/lon to x/y map projection coordinates in meters
    tri_pt = mtri.Triangulation(xpt, ypt, tri_data)
    # min_circle_ratio = 0.1
    # mask = TriAnalyzer(tri_pt).get_flat_tri_mask(min_circle_ratio)
    # tri_pt.set_mask(mask)
    triangles = data_dict['triangles']

    #bottom_depth = np.array(data_dict['depth_in_m'])  # as single number array
    var_name = 'grid_depth_' + depth
    var = np.array(data_dict[var_name])


    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.drawrivers()

    # Draw depth on the map using triangulation or gridded data
    # color_map = plt.cm.get_cmap('Blues_r')
    # color_map_r = color_map.reversed()
    #cax = plt.tripcolor(xpt, ypt, triangles, var, cmap='YlOrBr', edgecolors= 'none')
    cax = plt.tripcolor(xpt, ypt, triangles, var, cmap='Blues', edgecolors='none', vmin=np.nanmin(var), vmax=np.nanmax(var))

    #cax = plt.tripcolor(xpt, ypt, triangles, -depth, cmap='Blues_r', edgecolors=edge_color, vmin=-5000, vmax=0)

    # set the nan to white on the map
    #masked_array = np.ma.array(var, mask=np.isnan(var)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w')
    #cax = plt.tripcolor(xpt, ypt, triangles, masked_array, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var), vmax=np.nanmax(var))

    cbar = fig.colorbar(cax, shrink=0.7) #set scale bar
    cbar.set_label('Salinity [psu]', size=14) #scale label
    # labels = [left,right,top,bottom]
    parallels = np.arange(bot_lat, top_lat, 4.) # parallels = np.arange(48., 54, 0.2); parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  #draw parallel lat lines
    meridians = np.arange(left_lon, -100.0, 15.) # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])
    plt.show()
    png_name = os.path.join(file_path, output_folder + '/S_' + season + '_tri_' + depth + '.png')
    #fig.savefig(png_name, dpi=400)


#-----------------------------------------------------------------------------------------------------------------------
#set boundary

# left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]   # NE Paicif
# left_lon, right_lon, bot_lat, top_lat = [-140, -120, 45, 56]   # EEZ

def triangle_to_regular(data_dict, file_path, left_lon, right_lon, bot_lat, top_lat, output_folder, season, depth):
     tri_filename = os.path.join(file_path, 'nep35_reord.tri')
     tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    #build regular grid mesh and interpolate value on to the regular mesh
     # print(data_dict['y_lat'].max(), data_dict['y_lat'].min())
     # print(data_dict['x_lon'].max(), data_dict['x_lon'].min())
     #xi = np.linspace(clim_data['x_lon'].min(), clim_data['x_lon'].max(), 4422)   # ~ 1000m ~ 0.01 degree, for full NE Pacific
     #yi = np.linspace(clim_data['y_lat'].min(), clim_data['y_lat'].max(), 3151)   # ~ 1000m ~ 0.01 degree, for full NE Pacific
     xi = np.linspace(221, 239, 5400)  # ~ 333m ~ 0.003 degree
     yi = np.linspace(46, 55, 2700)  # ~ 333m ~ 0.003 degree
     x_lon_r, y_lat_r = np.meshgrid(xi, yi)  # create regular grid

     # create basemap
     m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                 urcrnrlon=right_lon, urcrnrlat=top_lat,
                 projection='lcc',  # width=40000, height=40000, #lambert conformal project
                 resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                 lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

     xpr, ypr = m(x_lon_r, y_lat_r) #convert lat/lon to x/y map projection coordinates in meters using basemap

     #2nd method to convert lat/lon to x/y
     #import pyproj
     #proj_basemap = pyproj.Proj(m.proj4string) # find out the basemap projection
     #t_lon, t_lat = proj_basemap(x_lon_g, y_lat_g)

     #get triangular mesh information
     x_lon = data_dict['x_lon']
     y_lat = data_dict['y_lat']
     xpt, ypt = m(x_lon, y_lat)  # convert lat/lon to x/y map projection coordinates in meters
     tri_pt = mtri.Triangulation(xpt, ypt, tri_data)
     trifinder = tri_pt.get_trifinder()  # trifinder= mtri.Triangulation.get_trifinder(tri_pt), return the default of this triangulation

     var_name = 'grid_depth_' + depth
     var = np.array(data_dict[var_name])

     # interpolate from triangular to regular mesh
     interp_lin = mtri.LinearTriInterpolator(tri_pt, var, trifinder=None) #conduct interpolation on lcc projection, not on lat/long
     var_r = interp_lin(xpr, ypr)
     var_r[var_r.mask] = np.nan # set the value of masked point to nan


     fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
     #m.drawcoastlines(linewidth=0.2)
     #m.drawmapboundary(fill_color='white')
     #m.fillcontinents(color='0.8')
     #m.scatter(xpr, ypr, color='black')
     cax = plt.pcolor(xpr, ypr, var_r, cmap='Blues', edgecolors= 'none')
     #cax = plt.pcolor(xpt, ypt, var_r, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var_r), vmax=np.nanmax(var_r))

     # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
     color_map = plt.cm.get_cmap()
     color_map.set_bad('w') #set the nan values to white on the plot


     cbar = fig.colorbar(cax, shrink=0.7) #set scale bar
     cbar.set_label('Salinity [psu]', size=14) #scale label
     parallels = np.arange(bot_lat-1, top_lat+1, 3.) #  parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
     m.drawparallels(parallels, labels=[True, False, False, False])  #draw parallel lat lines
     meridians = np.arange(-140, -120.0, 5.) # meridians = np.linspace(int(left_lon), right_lon, 5)
     m.drawmeridians(meridians, labels=[False, False, True, True])
     # labels = [left,right,top,bottom]
     plt.show()
     png_name = os.path.join(file_path, output_folder + '/S_' + season + '_reg_' + depth + '.png')
     #fig.savefig(png_name, dpi=400)

     # save the lat, lon and var on regular grid
     data_dict_new = dict()
     data_dict_new['x_lon_r'] = x_lon_r - 360
     data_dict_new['y_lat_r'] = y_lat_r
     data_dict_new['y_lat_r'] = y_lat_r
     data_dict_new[var_name ] = var_r
     return data_dict_new


#-----------------------------------------------------------------------------------------------------------------------
# Write interpolated data into geoTiff file
import rasterio
import pyproj
from rasterio.transform import Affine

#latlon = '+proj=longlat +datum=WGS84'
#proj_basemap = pyproj.Proj(m.proj4string) # find out the basemap projection

def convert_to_tif(data_dict, file_path, output_folder, season, depth):
    var_name = 'grid_depth_' + depth
    x_lon_r = data_dict['x_lon_r']
    y_lat_r = data_dict['y_lat_r']
    var_r = data_dict[var_name]
    res = (x_lon_r[0][-1] - x_lon_r[0][0])/5400
    transform = Affine.translation(x_lon_r[0][0] -res/2, y_lat_r[0][0] -res/2) * Affine.scale(res, res)

    tif_name = os.path.join(file_path, output_folder + '/S_' + season + '_' + depth + '.tif')

    raster_output = rasterio.open(
        tif_name,
        'w',
        driver='GTiff',
        height=var_r.shape[0],
        width=var_r.shape[1],
        count=1,
        dtype= var_r.dtype,
        #crs='+proj=longlat +datum=WGS84',
        crs='epsg:4326', #crs='+proj=latlong', #epsg:4326, Proj4js.defs["EPSG:4326"] = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        transform=transform,
        nodata = 0
    )

    raster_output.write(var_r.data, 1)
    raster_output.close()

#------------------------- fit into EEZ polygon shapefile------------------------
EEZ_clip(file_path = '/home/guanl/Desktop/MSP/Climatology', output_folder = 'S_sum', season = 'sum', depth = '0')

def EEZ_clip(file_path, output_folder, season, depth):
    tif_name = os.path.join(file_path, output_folder + '/S_' + season + '_' + depth + '.tif')
    tif_name_mask = os.path.join(file_path, output_folder + '/S_' + season + '_' + depth + '_masked.tif')
    with fiona.open ("/home/guanl/Desktop/MSP/Shapefiles/BC_EEZ/BC_EEZ/bc_eez.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(tif_name) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop = True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(tif_name_mask, "w", **out_meta) as dest:
        dest.write(out_image)



import numpy as np

mat_const= np.zeros(1, dtype={'names':['tMelt','k','alpha','delta_h','name'], 'formats':['f8','f8','f8','f8','a30']})

mat_const[0] = (1727.,  34.1,   5.18e-6,   9.41e9, "304 stainless")
mat_const = np.append(mat_const, np.array([(1713.,  60.45,  12.25e-6,  6.992e9, "15-5PH stainless")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1713.,  74.663, 15.548e-6, 6.804e9, "17-4 stainless")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1803.,  38.0,   5.5e-6,    10.4e9, "1018 steel")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1769.,  38.0,   5.5e-6,    10.4e9, "HY130 steel")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1776.,  44.0,   6.2e-6,    10.4e9, "HY80 steel")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(505.,   64.6,   17.3e-6,   0.78e9, "tin")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(2894.,  34.82,  8.3e-6,    10.9e9, "molybdenum")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1726.,  93.223, 11.338e-6, 11.766e9, "nickel 200")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1763.,  86.899, 14.457e-6, 8.818e9, "kovar")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(933.,   276.61, 84.80e-6,  2.045e9, "1100 aluminum")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(923.,   331.84, 147.31e-6, 1.412e9, "6061 aluminum")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1356.,  470.2,  93.384e-6, 5.337e9, "110 copper")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1670.,  103.11, 19.75e-6,  7.173e9, "hastelloy C4")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1672.,  95.086, 20.00e-6,  6.54e9, "hastelloy C22")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1704.,  82.45,  15.521e-6,  7.479e9, "hastelloy B2")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1533.,  17.88,   4.33e-6,   7.14e9, "Inconel 718")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1622.,  65.7,   13.0e-6 ,  7.12e9, "Inconel 625")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1877.,  12.8, 5.52E-06,   4.481e9, "Ti-6Al-4V")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1255.+ 273, 29.2900, 9.87200e-06, 1., "U6Nb")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1588., 24.3, 5.23707E-06, 1., "H13")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(3037.+273., 5.44e+01, 2.38806e-05, 1.,"Ta10W (unverified)")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1400.+273., 21.5,  5.38e-6, 1., "316 stainless")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(619.+273., 120,  45.e-6, 1., "Pandalloy")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(3695., 89.500,  2.0919e-05, 1., "Tungsten (unverified)")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1255.+ 273, 45.00, 1.34e-5, 1., "U6Nb (based on U-10Mo)")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(936, 260., 53.e-6, 1., "Aluminum Alloy A20x (unverified)")],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1703, 70.4, 1.24e-5, 1., "NiNb5" )],dtype = mat_const.dtype))
mat_const = np.append(mat_const, np.array([(1644, 21.5, 4.54e-6, 1., "HEA")],dtype = mat_const.dtype))

# To add extras, just copy the last line and change data values


"""Strong quasar emission lines (rest vacuum wavelengths, Vanden Berk et al. 2001, AJ 122, 549, Table 2) and the
SPHEREx wavelength range."""

# Strong, well-separated lines that at some redshift fall in the SPHEREx window (0.75-5 um); they carry essentially
# all the redshift information at R~40. (label, lab wavelength [A], rest equivalent width W [A] of the composite).
STRONG_LINES = [
    ("Lya",  1215.67,  92.91),
    ("CIV",  1549.06,  23.78),
    ("CIII", 1908.73,  21.19),
    ("MgII", 2798.75,  32.28),
    ("Hb",   4862.68,  46.21),
    ("OIII", 5008.24,  13.23),
    ("Ha",   6564.61, 194.52),
]

# SPHEREx coverage; the upper edge leaves room for measured channel centres up to ~5.01 um.
SPHEREX_MIN_UM = 0.75
SPHEREX_MAX_UM = 5.05

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
from numpy.typing import NDArray


def lumdens_to_sb(
    shape,
    ranges: NDArray[np.float64] | None = None,
    boxsize_pkpc: float | None = None,
    z: float = 2.0,
    cosmology: FlatLambdaCDM = Planck15,
    los="x",
):
    """Convert the luminosity density (erg/s/pMpc^3) to surface brightness (erg/s/cm^2/arcsec^2)
    for a single cube voxel."""
    assert los == "x", "Only x-axis LOS supported here for simplicity."
    if ranges is None:
        # assert we cover the whole box
        ranges = np.zeros((3, 2))
        ranges[:, 1] = 1.0
    if boxsize_pkpc is None:
        # assume TNG50 boxsize
        boxsize_pkpc = 35000.0 / Planck15.h / (1 + z)
    vol_cube_pkpc3 = boxsize_pkpc**3 * np.prod(np.diff(ranges, axis=1), axis=0)
    vol_voxel_pkpc3 = np.squeeze(vol_cube_pkpc3 / np.prod(shape, axis=0))
    area_cube_pkpc2 = boxsize_pkpc**2 * (
        (ranges[1, 1] - ranges[1, 0]) * (ranges[2, 1] - ranges[2, 0])
    )
    area_voxel_pkpc2 = np.squeeze(area_cube_pkpc2 / np.prod(shape[1:], axis=0))
    ldens_factor = get_units(
        "luminosity_density", cosmology, z, volume=vol_voxel_pkpc3 * u.kpc**3
    )
    sb_factor = get_units(
        "surface_brightness",
        cosmology,
        z,
        area=area_voxel_pkpc2 * u.kpc**2,
        codelength=1.0,
    )
    factor = (sb_factor / ldens_factor).to(u.Mpc**3 / (u.cm**2 * u.arcsec**2))
    return factor


def get_units(
    ustr: str,
    cosmology: FlatLambdaCDM,
    z: float,
    area: u.Quantity | float | None = None,
    volume: u.Quantity | float | None = None,
    codelength: u.Quantity | float | None = None,
):
    """Returns desired units from code units (1e42 erg/s)."""
    if ustr == "luminosity":
        # Each photon carries that amount of power times weight.
        units = 1.0e42 * u.erg / u.s
    elif ustr == "flux":
        lum = get_units("luminosity", cosmology, z)
        d = cosmology.luminosity_distance(z).to(u.cm)
        units = lum / (4 * np.pi * d**2)
    elif ustr == "surface_brightness":
        assert z != 0, "Surface brightness not defined at z=0.0"
        da_element = cosmology.angular_diameter_distance(z)
        flux = get_units("flux", cosmology, z)
        if not (type(area) is u.quantity.Quantity):  # code units to distance
            if codelength is None:
                raise ValueError("Need boxsize for SB units.")
            if area is None:
                raise ValueError("Need area for SB units.")
            area = area * codelength**2
        # squared angle observed on sky (pixel size over angular distance)
        alpha2 = ((area / da_element**2) * u.rad**2).to(u.arcsec**2)
        units = flux / alpha2
    elif ustr == "luminosity_density":
        lum = get_units("luminosity", cosmology, z)
        return lum / volume
    else:
        raise ValueError("Unknown units.")
    return units

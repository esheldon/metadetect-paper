import images
import biggles
import galsim

# arcsec/pixel
SCALE=0.01
SHEAR=(0.1, 0.0)
LEVELS=2
FWHM=0.47
ccolor='blue'

def get_objects():
    o1 = galsim.Gaussian(fwhm=FWHM)
    o2 = galsim.Gaussian(fwhm=FWHM)

    shift=0.4
    o1 = o1.shift(dx=-shift, dy=0.0)
    o2 = o2.shift(dx=+shift, dy=0.0)

    return o1,o2

def get_psf():
    return galsim.Gaussian(fwhm=FWHM)

def get_image(objs):
    return objs.drawImage(
        nx=200,
        ny=200,
        scale=SCALE,
    ).array

def main():

    o1, o2 = get_objects()
    psf = get_psf()

    allo = galsim.Add(o1, o2)
    allo_sheared = allo.shear(
        g1=SHEAR[0],
        g2=SHEAR[1],
    )

    pallo = galsim.Convolve(allo, psf)
    pallo_sheared = galsim.Convolve(allo_sheared, psf)

    im = get_image(allo)
    im_sheared = get_image(allo_sheared)

    pim = get_image(pallo)
    pim_sheared = get_image(pallo_sheared)

    plt=images.view(
        im,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        title='shear: 0, 0',
        show=False,
    )

    plt_sheared=images.view(
        im_sheared,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        title='shear: %g, %g' % SHEAR,
        show=False,
    )

    pplt=images.view(
        pim,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        title='shear: 0, 0   with PSF',
        show=False,
    )

    pplt_sheared = images.view(
        pim_sheared,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        title='shear: %g, %g   with PSF' % SHEAR,
        show=False,
    )

    tab=biggles.Table(2,2,aspect_ratio=1)

    tab[0,0] = plt
    tab[0,1] = plt_sheared
    tab[1,0] = pplt
    tab[1,1] = pplt_sheared
    tab.show()
    tab.write('toy.pdf')

main()

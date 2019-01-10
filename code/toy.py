import numpy as np
import images
import biggles
import galsim
from galsim import Convolve, Add

# arcsec/pixel
SCALE=0.01
SHEAR=(0.1, 0.0)
LEVELS=2
ZPERC=[1, 100.0]
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
    im = objs.drawImage(
        nx=200,
        ny=200,
        scale=SCALE,
    ).array

    im *= 1.0/im.max()
    return im

def main():

    o1, o2 = get_objects()
    psf = get_psf()

    allo_pre = Convolve(
        Add(o1, o2),
        psf,
    )
    allo_pre_sheared = allo_pre.shear(
        g1=SHEAR[0],
        g2=SHEAR[1],
    )

    allo_post_sheared = Convolve(Add(o1, o2).shear(g1=SHEAR[0],g2=SHEAR[1]), psf)

    im_pre = get_image(allo_pre)
    im_pre_sheared = get_image(allo_pre_sheared)
    im_post_sheared = get_image(allo_post_sheared)

    plt_pre=images.view(
        im_pre,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        zrange=np.percentile(im_pre, ZPERC),
        #title='shear: 0, 0',
        show=False,
    )

    plt_pre_sheared=images.view(
        im_pre_sheared,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        zrange=np.percentile(im_pre_sheared, ZPERC),
        #title='shear: %g, %g' % SHEAR,
        show=False,
    )

    plt_post_sheared=images.view(
        im_post_sheared,
        type='dens-cont',
        ccolor=ccolor,
        levels=LEVELS,
        zrange=np.percentile(im_post_sheared, ZPERC),
        #title='shear: 0, 0   with PSF',
        show=False,
    )

    plt_pre.add( 
        biggles.PlotLabel(0.9, 0.9, '(a)', halign='right', color='white')
    )
    plt_pre_sheared.add( 
        biggles.PlotLabel(0.9, 0.9, '(b)', halign='right', color='white')
    )
    plt_post_sheared.add( 
        biggles.PlotLabel(0.9, 0.9, '(c)', halign='right', color='white')
    )

    """
    tab=biggles.Table(3,1,aspect_ratio=3.0)

    tab[0,0] = plt_pre
    tab[1,0] = plt_pre_sheared
    tab[2,0] = plt_post_sheared
    """

    tab=biggles.Table(1,3,aspect_ratio=1/3)

    tab[0,0] = plt_pre
    tab[0,1] = plt_pre_sheared
    tab[0,2] = plt_post_sheared

    tab.show()
    tab.write('toy.pdf')

main()

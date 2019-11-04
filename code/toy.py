import numpy as np
import images
import biggles
import galsim
import argparse

# arcsec/pixel
SCALE = 0.01
SHEAR = (0.1, 0.0)
LEVELS = 2
ZPERC = [1, 100.0]
FWHM = 0.47
CCOLOR = 'grey65'
SHIFT = 0.4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-full-scene', action='store_true')
    return parser.parse_args()


def get_objects(args):
    o1 = galsim.Gaussian(fwhm=FWHM)
    o2 = galsim.Gaussian(fwhm=FWHM)

    if not args.no_full_scene:
        o1 = o1.shift(dx=-SHIFT, dy=0.0)
        o2 = o2.shift(dx=+SHIFT, dy=0.0)

    return o1, o2


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

    args = get_args()

    o1, o2 = get_objects(args)
    psf = get_psf()

    if args.no_full_scene:
        name = 'toy-no-full-scene'

        allo = galsim.Add(
            o1.shift(dx=-SHIFT, dy=0.0),
            o2.shift(dx=+SHIFT, dy=0.0),
        )
        allo_pre = galsim.Convolve(
            allo,
            psf,
        )


        o1_pre = galsim.Convolve(o1, psf).shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )
        o2_pre = galsim.Convolve(o2, psf).shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )
        o1_pre = o1_pre.shift(dx=-SHIFT, dy=0.0)
        o2_pre = o2_pre.shift(dx=+SHIFT, dy=0.0)
        allo_pre_sheared = galsim.Add(o1_pre, o2_pre)

        o1_post = o1.shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )
        o2_post = o2.shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )
        o1_post = galsim.Convolve(o1_post, psf)
        o2_post = galsim.Convolve(o2_post, psf)

        o1_post = o1_post.shift(dx=-SHIFT, dy=0.0)
        o2_post = o2_post.shift(dx=+SHIFT, dy=0.0)

        allo_post_sheared = galsim.Add(o1_post, o2_post)

    else:
        name = 'toy'

        allo = galsim.Add(o1, o2)
        allo_pre = galsim.Convolve(
            allo,
            psf,
        )


        allo_pre = galsim.Convolve(
            allo,
            psf,
        )
        allo_pre_sheared = allo_pre.shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )

        allo_post = galsim.Add(o1, o2).shear(
            g1=SHEAR[0],
            g2=SHEAR[1],
        )
        allo_post_sheared = galsim.Convolve(
            allo_post,
            psf,
        )

    im_pre = get_image(allo_pre)
    im_pre_sheared = get_image(allo_pre_sheared)
    im_post_sheared = get_image(allo_post_sheared)

    plt_pre = images.view(
        im_pre,
        type='dens-cont',
        ccolor=CCOLOR,
        levels=LEVELS,
        zrange=np.percentile(im_pre, ZPERC),
        invert=True,
        show=False,
    )

    plt_pre_sheared = images.view(
        im_pre_sheared,
        type='dens-cont',
        ccolor=CCOLOR,
        levels=LEVELS,
        zrange=np.percentile(im_pre_sheared, ZPERC),
        invert=True,
        show=False,
    )

    plt_post_sheared = images.view(
        im_post_sheared,
        type='dens-cont',
        ccolor=CCOLOR,
        levels=LEVELS,
        zrange=np.percentile(im_post_sheared, ZPERC),
        invert=True,
        show=False,
    )

    plt_pre.add(
        biggles.PlotLabel(0.9, 0.9, '(a)', halign='right', color='black')
    )
    plt_pre_sheared.add(
        biggles.PlotLabel(0.9, 0.9, '(b)', halign='right', color='black')
    )
    plt_post_sheared.add(
        biggles.PlotLabel(0.9, 0.9, '(c)', halign='right', color='black')
    )

    tab = biggles.Table(1, 3, aspect_ratio=1/3)

    tab[0, 0] = plt_pre
    tab[0, 1] = plt_pre_sheared
    tab[0, 2] = plt_post_sheared

    tab.show()
    tab.write('%s.pdf' % name)
    tab.write('%s.png' % name, dpi=150)


if __name__ == '__main__':
    main()

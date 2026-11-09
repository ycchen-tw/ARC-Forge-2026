# arms.sh (sourced by serve.sh and run_arm.sh): arm name -> draft directory under $MTPD_RUN/drafts.
#   moe320          mtp320-fp8            (MTP MoE pruned to the target's 320 experts)
#   moe320-<x>      mtp320-fp8-<x>        (L<n>: that layer stacked n deep, bit copies; a layer spec such as M_T43_T47:
#                                          the MTP layer then target layers 43, 47, build_drafts.py --spec M,T43,T47;
#                                          T47 / T47m: the 10-11 names of M_T47 [with the target's final mixer])
#   dense-ls        dense10240-ls         dense-uniform  dense10240-uniform
#   dense-ls-<x>    dense10240-ls-<x>     (every layer of spec <x> densified, build_drafts.py dense --spec)
#   dense<W>-ls-<x> dense<W>-ls-<x>       (width W = experts x 640, width_sweep.sh)
#   collect         mtp320-fp8            (eager, MoE input sampling to $MTPD_RUN/collect/rows)
#   collect-<x>     mtp320-fp8-<x>        (the same for every layer of that draft, to $MTPD_RUN/collect-<x>/rows)
arm_draft() {
    case $1 in
        collect|moe320) echo "$MTPD_RUN/drafts/mtp320-fp8" ;;
        collect-*) echo "$MTPD_RUN/drafts/mtp320-fp8-${1#collect-}" ;;
        moe320-*) echo "$MTPD_RUN/drafts/mtp320-fp8-${1#moe320-}" ;;
        dense-ls|dense-uniform|dense-ls-*|dense-uniform-*) echo "$MTPD_RUN/drafts/dense10240-${1#dense-}" ;;
        dense[0-9]*-ls-*|dense[0-9]*-uniform-*) echo "$MTPD_RUN/drafts/$1" ;;  # other widths (width_sweep.sh)
        *) echo "✗ unknown arm $1" >&2; return 1 ;;
    esac
}
# collect arms: where the collector writes ("" for the others)
arm_collect_dir() {
    case $1 in
        collect) echo "$MTPD_RUN/collect/rows" ;;
        collect-*) echo "$MTPD_RUN/$1/rows" ;;
        *) echo "" ;;
    esac
}

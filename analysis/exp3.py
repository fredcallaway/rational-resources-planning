%reset -f
%run -i setup 3
figs.nosave = True
%run -i model_comparison
%run -i breadth_depth
%run -i expansion
figs.nosave = False

@figure(tight=False)
def exp3_main():
    fig, axes = plt.subplots(2, 3, figsize=(11.9, 0.9 * 6), 
        constrained_layout=True,
        gridspec_kw=dict(height_ratios=[1, 1.2]))
    for v, ax in zip(VARIANCES, axes[0, :]):
        ax.set_title(f'{v.title()} Variance', fontdict=dict(fontsize=20), pad=20)

    first_click_depth(axes[0])
    fig.set_constrained_layout_pads(h_pad=0.1)
    plot_geometric_mean_likelihood(axes[1])

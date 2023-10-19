class design:
    print_figure_title = True

    title_font_size = 18
    legend_font_size = 16

    legend_x_pos = 0
    legend_y_pos = 1

    scatter_legend = dict(
        x=legend_x_pos,
        y=legend_y_pos,
        orientation="h",
        traceorder="normal",
        font=dict(
            size=legend_font_size,
        ),
    )
    base_theme = "simple_white"

    showgrid = False

    scatter_markers = "lines"

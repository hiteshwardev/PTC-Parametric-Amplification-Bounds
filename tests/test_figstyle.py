import numpy as np

import figstyle as fs


def _figure():
    fig, ax = fs.new_figure(9.0)
    x = np.linspace(0.0, 1.0, 40)
    ax.plot(x, x ** 2, **fs.style(0))
    ax.set_xlabel(r"wavevector $k c/\omega_{r0}$")
    ax.set_ylabel(r"exponent $\mu/\omega_{r0}$")
    fs.legend_below(fig, [fs.line_handle("series", **fs.style(0))], ncol=1)
    return fig, ax


def test_clean_figure_passes():
    fig, _ = _figure()
    assert fs.layout_problems(fig) == []


def test_text_inside_plot_is_rejected():
    fig, ax = _figure()
    ax.text(0.5, 0.5, "note")
    assert any("inside the plot" in p for p in fs.layout_problems(fig))


def test_plain_text_subscript_is_rejected():
    fig, ax = _figure()
    ax.set_xlabel("omega_r")
    assert any("unformatted symbol" in p for p in fs.layout_problems(fig))

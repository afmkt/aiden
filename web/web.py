import solara

x = solara.reactive(2)


@solara.component
def Page():
    x_squared = x.value**2

    with solara.Sidebar():
        solara.Markdown("## My First Solara app ☀️")
        solara.SliderInt(label="x", value=x)
    solara.Markdown(f"{x.value} squared = {x_squared}")
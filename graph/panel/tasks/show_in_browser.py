import numpy as np
import hvplot.pandas  # noqa
import pandas as pd
import panel as pn

# Create some data
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# Create a hvplot
plot = df.hvplot.line()

# Create a Panel app
app = pn.Column('# Hello HvPlot', plot)

# Run the app
app.show()
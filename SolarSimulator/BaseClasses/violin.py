import plotly.express as px
import pandas as pd
import numpy as np

# Create a sample DataFrame
np.random.seed(10)
df = pd.DataFrame({
    "Category": np.random.choice(["A", "B", "C"], 100),
    "Value": np.random.randn(100)
})

# Create a violin plot
fig = px.violin(df, x="Category", y="Value", box=True, points="all", title="Violin Plot Example")

# Show the plot
fig.show()

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def test_cmd():
    pass


if __name__ == "__main__":
    app()

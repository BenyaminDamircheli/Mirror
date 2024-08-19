engine = PhotoEngine("default")
opened_image = engine.open_image("https://plus.unsplash.com/premium_photo-1670745084868-7b4f727cc934?q=80&w=2864&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
print(engine.describe_image(opened_image))
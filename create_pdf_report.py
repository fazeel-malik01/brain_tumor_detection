from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import utils
import os

def add_image_to_canvas(c, image_path, x, y, width, height):
    if os.path.exists(image_path):
        img = utils.ImageReader(image_path)
        img_width, img_height = img.getSize()
        aspect_ratio = img_height / float(img_width)
        if width / height > aspect_ratio:
            new_width = height / aspect_ratio
            new_height = height
        else:
            new_width = width
            new_height = width * aspect_ratio
        
        c.drawImage(image_path, x, y, width=new_width, height=new_height)

def generate_pdf_report():
    c = canvas.Canvas("model_report.pdf", pagesize=letter)
    width, height = letter

    # Define vertical position for content
    y_position = height - 1 * inch
    margin = 1 * inch
    image_width = 6 * inch
    image_height = 3 * inch
    spacing = 0.5 * inch

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y_position, "Brain Tumor Classification Report")
    y_position -= 1 * inch

    # Add Accuracy and Loss Graphs
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, "Accuracy and Loss Graphs:")
    y_position -= spacing

    # Accuracy Graph
    add_image_to_canvas(c, "Reports/training_accuracy.png", margin, y_position - image_height, image_width, image_height)
    y_position -= image_height + spacing

    # Check for page overflow
    if y_position < 2 * inch:
        c.showPage()  # Create a new page
        y_position = height - 1 * inch

    # Loss Graph
    add_image_to_canvas(c, "Reports/training_loss.png", margin, y_position - image_height, image_width, image_height)
    y_position -= image_height + spacing

    # Check for page overflow
    if y_position < 2 * inch:
        c.showPage()  # Create a new page
        y_position = height - 1 * inch

    # Add Confusion Matrix
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, "Confusion Matrix:")
    y_position -= spacing

    # Confusion Matrix Graph
    add_image_to_canvas(c, "Reports/confusion_matrix.png", margin, y_position - image_height, image_width, image_height)
    y_position -= image_height + spacing

    # Check for page overflow
    if y_position < 2 * inch:
        c.showPage()  # Create a new page
        y_position = height - 1 * inch

    # Add Classification Report
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, "Classification Report:")
    y_position -= spacing

    # Check if the Classification Report fits on the current page
    if y_position < 2 * inch:
        c.showPage()  # Create a new page
        y_position = height - 1 * inch

    # Add Classification Report image
    add_image_to_canvas(c, "Reports/classification_report.png", margin, y_position - image_height, image_width, image_height)

    # Save the PDF
    c.save()

generate_pdf_report()

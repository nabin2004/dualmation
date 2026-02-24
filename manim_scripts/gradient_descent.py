"""
Example Manim scene: Gradient Descent Visualization

This is a sample scene that DualAnimate's LLM brain would generate
for the concept "Explain gradient descent visually".
"""

from manim import *


class GradientDescentScene(Scene):
    """Visualizes the gradient descent optimization process."""

    def construct(self):
        # Title
        title = Text("Gradient Descent", font_size=48, color=BLUE)
        subtitle = Text("Finding the minimum of a function", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Create axes and function
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=8,
            y_length=5,
            axis_config={"color": GRAY},
        )
        labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        # Quadratic function: f(x) = x^2 + 1
        func = axes.plot(lambda x: x**2 + 1, color=BLUE, x_range=[-2.8, 2.8])
        func_label = MathTex(r"f(x) = x^2 + 1", color=BLUE).to_corner(UR)

        self.play(Create(axes), Write(labels))
        self.play(Create(func), Write(func_label))
        self.wait(0.5)

        # Gradient descent steps
        x_val = ValueTracker(2.5)
        learning_rate = 0.3

        dot = always_redraw(
            lambda: Dot(
                axes.c2p(x_val.get_value(), x_val.get_value() ** 2 + 1),
                color=RED,
                radius=0.1,
            )
        )

        # Tangent line
        tangent = always_redraw(
            lambda: axes.plot(
                lambda x: 2 * x_val.get_value() * (x - x_val.get_value())
                + x_val.get_value() ** 2 + 1,
                color=YELLOW,
                x_range=[
                    x_val.get_value() - 1,
                    x_val.get_value() + 1,
                ],
            )
        )

        # Step label
        step_label = always_redraw(
            lambda: MathTex(
                f"x = {x_val.get_value():.2f}",
                color=RED,
            ).next_to(dot, UP + RIGHT, buff=0.2)
        )

        self.play(Create(dot), Create(tangent), Write(step_label))

        # Gradient descent iterations
        gradient_text = Text("Gradient Descent Steps", font_size=28, color=GREEN)
        gradient_text.to_corner(UL)
        self.play(Write(gradient_text))

        for i in range(6):
            current_x = x_val.get_value()
            gradient = 2 * current_x  # derivative of x^2 + 1
            new_x = current_x - learning_rate * gradient

            step_info = MathTex(
                f"\\text{{Step {i+1}}}: \\nabla f = {gradient:.2f}",
                font_size=24,
                color=YELLOW,
            )
            step_info.next_to(gradient_text, DOWN, buff=0.3 + i * 0.35)

            self.play(
                x_val.animate.set_value(new_x),
                Write(step_info),
                run_time=0.8,
            )
            self.wait(0.3)

        # Final position
        converged = Text("Converged! ✓", font_size=32, color=GREEN)
        converged.next_to(axes, DOWN, buff=0.5)
        self.play(Write(converged))
        self.wait(2)

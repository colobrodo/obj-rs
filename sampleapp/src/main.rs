use glium::backend::glutin::SimpleWindowBuilder;
use glium::backend::Facade;
use glium::glutin::surface::WindowSurface;
use glium::uniforms::{AsUniformValue, Uniforms, UniformsStorage};
use glium::winit::application::ApplicationHandler;
use glium::winit::event::WindowEvent;
use glium::winit::event_loop::{ControlFlow, EventLoop};
use glium::winit::window::Window;
use glium::{uniform, Display, IndexBuffer, Program, Surface, VertexBuffer};
use obj::{load_obj, Obj, Vertex};

struct AppState<'a, U: AsUniformValue, IS: Uniforms> {
    display: Display<WindowSurface>,
    window: Window,
    vb: VertexBuffer<Vertex>,
    ib: IndexBuffer<u16>,
    program: Program,
    uniforms: UniformsStorage<'a, U, IS>,
    params: glium::DrawParameters<'a>,
}

struct SampleApp<'a, U: AsUniformValue, IS: Uniforms> {
    state: Option<AppState<'a, U, IS>>,
    obj: Obj<Vertex>,
}

impl<'a, U: AsUniformValue, IS: Uniforms> SampleApp<'a, U, IS> {
    fn new(obj: Obj<Vertex>) -> Self {
        SampleApp { state: None, obj }
    }
}

impl<'a, U: AsUniformValue, IS: Uniforms> ApplicationHandler for SampleApp<'a, U, IS> {
    fn resumed(&mut self, event_loop: &glium::winit::event_loop::ActiveEventLoop) {
        // building the display, ie. the main object
        let (window, display) = SimpleWindowBuilder::new()
            .with_inner_size(500, 400)
            .with_title("obj-rs")
            .build(event_loop);

        let vb = self.obj.vertex_buffer(display.get_context()).unwrap();
        let ib = self.obj.index_buffer(display.get_context()).unwrap();

        let program = Program::from_source(
            &display,
            r#"
                #version 410

                uniform mat4 matrix;

                in vec3 position;
                in vec3 normal;

                smooth out vec3 _normal;

                void main() {
                    gl_Position = matrix * vec4(position, 1.0);
                    _normal = normalize(normal);
                }
            "#,
            r#"
                #version 410

                uniform vec3 light;

                smooth in vec3 _normal;
                out vec4 result;

                void main() {
                    result = vec4(clamp(dot(_normal, -light), 0.0f, 1.0f) * vec3(1.0f, 0.93f, 0.56f), 1.0f);
                }
            "#,
            None,
        )
        .expect("Error during the compilation of the shaders");

        // drawing a frame
        let uniforms = uniform! {
            matrix: [
                [ 2.356724, 0.000000, -0.217148, -0.216930],
                [ 0.000000, 2.414214,  0.000000,  0.000000],
                [-0.523716, 0.000000, -0.977164, -0.976187],
                [ 0.000000, 0.000000,  9.128673,  9.219544f32]
            ],
            light: (-1.0, -1.0, -1.0f32)
        };

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };

        self.state = Some(AppState {
            display,
            window,
            vb,
            ib,
            uniforms,
            program,
            params,
        })
    }

    fn window_event(
        &mut self,
        event_loop: &glium::winit::event_loop::ActiveEventLoop,
        window_id: glium::winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // draw
                if let Some(state) = self.state {
                    let mut target = state.display.draw();
                    target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
                    target
                        .draw(
                            &state.vb,
                            &state.ib,
                            &state.program,
                            &state.uniforms,
                            &state.params,
                        )
                        .unwrap();
                    target.finish().unwrap();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;

    let input = include_bytes!("../../obj-rs/tests/fixtures/normal-cone.obj");
    let obj: Obj = load_obj(&input[..])?;

    let mut app = SampleApp::new(obj);
    event_loop.set_control_flow(ControlFlow::Wait);
    // Main loop
    event_loop.run_app(&mut app)?;

    Ok(())
}

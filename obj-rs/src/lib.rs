/*!

[Wavefront OBJ][obj] parser for Rust. It handles both `.obj` and `.mtl` formats. [GitHub][]

```rust
use std::fs::File;
use std::io::BufReader;
use obj::{load_obj, Obj};

let input = BufReader::new(File::open("tests/fixtures/normal-cone.obj")?);
let dome: Obj = load_obj(input)?;

// Do whatever you want
dome.vertices;
dome.indices;
# Ok::<(), obj::ObjError>(())
```

<img alt="Rendered image of cute Rilakkuma" src="https://i.hyeon.me/obj-rs/bear.png" style="max-width:100%">

[obj]: https://en.wikipedia.org/wiki/Wavefront_.obj_file
[GitHub]: https://github.com/simnalamburt/obj-rs

*/

#![deny(missing_docs)]

mod error;
pub mod raw;

pub use crate::error::{LoadError, LoadErrorKind, ObjError, ObjResult};

use crate::error::{index_out_of_range, make_error};
use crate::raw::object::Polygon;
use num_traits::{FromPrimitive, ToPrimitive};
use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::io::BufRead;

#[cfg(feature = "glium")]
use glium::implement_vertex;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "vulkano")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "vulkano")]
use vulkano::impl_vertex;

/// Load a wavefront OBJ file into Rust & OpenGL friendly format.
pub fn load_obj<V: FromRawVertex<I>, T: BufRead, I>(input: T) -> ObjResult<Obj<V, I>> {
    let raw = raw::parse_obj(input)?;
    Obj::new(raw)
}

/// Triangulate the polygon using the earcut algorithm
/// This function accept the vertex buffer and the indexes of the polygon to triangulate
fn triangulate<I: ToPrimitive + Copy>(polygon_indexes: &[I], vb: &[impl Point]) -> Vec<I> {
    assert!(polygon_indexes.len() >= 3);
    let mut result = Vec::new();
    let mut current_vertex_index = 0;
    // TODO: remove the clone
    let mut polygon = Vec::from(polygon_indexes);
    while polygon.len() > 3 {
        // take the vertex and the two adjacent ones
        let previous_index = (current_vertex_index + polygon.len() - 1) % polygon.len();
        let i0 = polygon[previous_index];
        let v0 = vb[i0.to_usize().unwrap()].position();
        let i1 = polygon[current_vertex_index];
        let v1 = vb[i1.to_usize().unwrap()].position();
        let next_index = (current_vertex_index + 1) % polygon.len();
        let i2 = polygon[next_index];
        let v2 = vb[i2.to_usize().unwrap()].position();
        // check if the angle is concave
        // to do so we check that the cross-product of their edges is positive if the polygon is in clockwise order
        // viceversa if counter-clockwise
        // to check the order we calculate the area using the shoelace formula and check if it is positive
        let cross_product = (v1[0] - v0[0]) * (v2[1] - v1[1]) - (v1[1] - v0[1]) * (v2[0] - v1[0]);
        let area = 0.5 * (v0[0] * v1[1] - v1[0] * v0[1]);
        if cross_product * area < 0.0 {
            // with reflex angles the triangle is an ear
            current_vertex_index = next_index;
            continue;
        }
        // check if this triangle contains any other vertex
        let mut index = (next_index + 1) % polygon.len();
        let mut contains_any_other_triangle = false;
        while index < previous_index {
            let i = polygon[index];
            let vertex_to_check = vb[i.to_usize().unwrap()].position();
            // to check if the vertex is inside the triangle the dot products of every triangle's side normal and the point have the same sign
            // to calculate the normal we simply rotate the side vector by 90 degrees (x, y) -> (-y, x)
            let v0_to_v1 = [v0[1] - v1[1], v1[0] - v0[0]];
            let v1_to_v2 = [v1[1] - v2[1], v2[0] - v1[0]];
            let v2_to_v0 = [v2[1] - v1[1], v0[0] - v2[0]];
            let dot_side0 = v0_to_v1[0] * vertex_to_check[0] + v0_to_v1[1] * vertex_to_check[1];
            let dot_side1 = v1_to_v2[0] * vertex_to_check[0] + v1_to_v2[1] * vertex_to_check[1];
            let dot_side2 = v2_to_v0[0] * vertex_to_check[0] + v2_to_v0[1] * vertex_to_check[1];
            if dot_side0 * dot_side1 > 0.0 && dot_side1 * dot_side2 > 0.0 {
                contains_any_other_triangle = true;
                break;
            }
            index = (index + 1) % polygon.len();
        }
        if contains_any_other_triangle {
            // if the triangle contains any other vertex is not an ear
            current_vertex_index = next_index;
            continue;
        }

        // otherwise it is an ear, add its vertex indexes to the result and remove the current vertice from the polygon
        // remove v1 from the polygon
        polygon.remove(current_vertex_index);
        // add the list of the three vertices to the result
        result.push(i0);
        result.push(i1);
        result.push(i2);
        // in case we remove the last vertex we should point now to the first, otherwise we point to the one previously indexed by `next_index`
        current_vertex_index = current_vertex_index % polygon.len();
    }
    // only three vertices left, add all of them to the result vertex indexes
    result.append(&mut polygon);
    result
}

/// 3D model object loaded from wavefront OBJ.
#[derive(Clone, Eq, PartialEq, Hash, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Obj<V = Vertex, I = u16> {
    /// Object's name.
    pub name: Option<String>,
    /// Vertex buffer.
    pub vertices: Vec<V>,
    /// Index buffer.
    pub indices: Vec<I>,
}

impl<V: FromRawVertex<I>, I> Obj<V, I> {
    /// Create `Obj` from `RawObj` object.
    pub fn new(raw: raw::RawObj) -> ObjResult<Self> {
        let (vertices, indices) =
            FromRawVertex::process(raw.positions, raw.normals, raw.tex_coords, raw.polygons)?;

        Ok(Obj {
            name: raw.name,
            vertices,
            indices,
        })
    }
}

/// Conversion from `RawObj`'s raw data.
pub trait FromRawVertex<I>: Sized {
    /// Build vertex and index buffer from raw object data.
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        normals: Vec<(f32, f32, f32)>,
        tex_coords: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)>;
}

/// Vertex data type of `Obj` which contains position and normal data of a vertex.
#[derive(Default, Copy, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "vulkano", repr(C))]
#[cfg_attr(feature = "vulkano", derive(Zeroable, Pod))]
pub struct Vertex {
    /// Position vector of a vertex.
    pub position: [f32; 3],
    /// Normal vertor of a vertex.
    pub normal: [f32; 3],
}

#[cfg(feature = "glium")]
implement_vertex!(Vertex, position, normal);
#[cfg(feature = "vulkano")]
impl_vertex!(Vertex, position, normal);

impl FromBufferIndexes<VertexIndex> for Vertex {
    fn from_buffer_index(
        positions: &[(f32, f32, f32, f32)],
        normals: &[(f32, f32, f32)],
        _: &[(f32, f32, f32)],
        vertex_index: VertexIndex,
    ) -> Self {
        let position = positions[vertex_index.pi];
        let normal = normals[vertex_index.ni];
        Vertex {
            position: [position.0, position.1, position.2],
            normal: [normal.0, normal.1, normal.2],
        }
    }
}

impl Point for Vertex {
    fn position(&self) -> [f32; 3] {
        [self.position[0], self.position[1], self.position[2]]
    }
}

/// Trait to extract the position of a vertex
pub trait Point {
    /// Return the position of the vertex
    fn position(&self) -> [f32; 3];
}

/// Construct the vertex type from a set of buffers and the indexes relative to them
pub trait FromBufferIndexes<VI> {
    /// Construct the vertex type from the buffers and the indexes
    fn from_buffer_index(
        positions: &[(f32, f32, f32, f32)],
        normals: &[(f32, f32, f32)],
        tex_coords: &[(f32, f32, f32)],
        vertex_index: VI,
    ) -> Self;
}

struct VertexBufferCache<
    I: ToPrimitive + FromPrimitive + Copy,
    VI: Eq + Hash + Copy = VertexIndex,
    V: FromBufferIndexes<VI> + Point = Vertex,
> {
    positions: Vec<(f32, f32, f32, f32)>,
    normals: Vec<(f32, f32, f32)>,
    tex_coords: Vec<(f32, f32, f32)>,
    cache: HashMap<VI, I>,
    vb: Vec<V>,
    ib: Vec<I>,
    phantom: std::marker::PhantomData<VI>,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct VertexIndex {
    pi: usize,
    ni: usize,
}

impl From<(usize, usize)> for VertexIndex {
    fn from((pi, ni): (usize, usize)) -> Self {
        VertexIndex { pi, ni }
    }
}

impl From<(usize, usize, usize)> for VertexIndex {
    fn from((pi, _, ni): (usize, usize, usize)) -> Self {
        VertexIndex { pi, ni }
    }
}

impl<
        I: ToPrimitive + FromPrimitive + Copy,
        VI: Eq + Hash + Copy,
        V: FromBufferIndexes<VI> + Point,
    > VertexBufferCache<I, VI, V>
{
    fn new(
        positions: Vec<(f32, f32, f32, f32)>,
        normals: Vec<(f32, f32, f32)>,
        tex_coords: Vec<(f32, f32, f32)>,
        n_polygons: usize,
    ) -> Self {
        VertexBufferCache {
            positions,
            normals,
            tex_coords,
            cache: HashMap::new(),
            vb: Vec::with_capacity(n_polygons * 3),
            ib: Vec::with_capacity(n_polygons * 3),
            phantom: std::marker::PhantomData,
        }
    }

    /// Given the indexes for a vertex, returns the vertex buffer index of the vertex they reference If it already exists.
    /// Otherwise creates a new vertex reading the buffers context and push the newly created vertex into the vertex buffer
    fn map(&mut self, buffer_indexes: VI) -> ObjResult<I> {
        // Look up cache
        let index = match self.cache.entry(buffer_indexes) {
            // Cache miss -> make new, store it on cache
            Entry::Vacant(entry) => {
                let vertex = V::from_buffer_index(
                    &self.positions,
                    &self.normals,
                    &self.tex_coords,
                    buffer_indexes,
                );
                let index = match I::from_usize(self.vb.len()) {
                    Some(val) => val,
                    None => return index_out_of_range::<_, I>(self.vb.len()),
                };
                self.vb.push(vertex);
                entry.insert(index);
                index
            }
            // Cache hit -> use it
            Entry::Occupied(entry) => *entry.get(),
        };
        Ok(index)
    }

    fn add_polygon(&mut self, vec: Vec<impl Into<VI>>) -> ObjResult<()> {
        let mut polygon_indexes = Vec::new();
        for vi in vec {
            let vi = vi.into();
            polygon_indexes.push(self.map(vi)?);
        }
        let vertices = triangulate(polygon_indexes.as_slice(), self.vb.as_slice());
        self.ib.extend(vertices);
        Ok(())
    }

    fn finalize(mut self) -> (Vec<V>, Vec<I>) {
        self.vb.shrink_to_fit();
        (self.vb, self.ib)
    }
}

impl<I: ToPrimitive + FromPrimitive + Copy> FromRawVertex<I> for Vertex {
    fn process(
        positions: Vec<(f32, f32, f32, f32)>,
        normals: Vec<(f32, f32, f32)>,
        tex_coord: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        let mut cache = VertexBufferCache::new(positions, normals, tex_coord, polygons.len());

        for polygon in polygons {
            match polygon {
                Polygon::P(_) | Polygon::PT(_) => make_error!(
                    InsufficientData,
                    "Tried to extract normal data which are not contained in the model"
                ),
                Polygon::PN(vec) => {
                    cache.add_polygon(vec)?;
                }
                Polygon::PTN(vec) => {
                    cache.add_polygon(vec)?;
                }
                _ => make_error!(
                    UntriangulatedModel,
                    "Model should be triangulated first to be loaded properly"
                ),
            }
        }
        Ok(cache.finalize())
    }
}

/// Vertex data type of `Obj` which contains only position data of a vertex.
#[derive(Default, Copy, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "vulkano", repr(C))]
#[cfg_attr(feature = "vulkano", derive(Zeroable, Pod))]
pub struct Position {
    /// Position vector of a vertex.
    pub position: [f32; 3],
}

#[cfg(feature = "glium")]
implement_vertex!(Position, position);
#[cfg(feature = "vulkano")]
impl_vertex!(Position, position);

impl<I: FromPrimitive> FromRawVertex<I> for Position {
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        let vb = vertices
            .into_iter()
            .map(|v| Position {
                position: [v.0, v.1, v.2],
            })
            .collect();
        let mut ib = Vec::with_capacity(polygons.len() * 3);
        {
            let mut map = |pi: usize| -> ObjResult<()> {
                ib.push(match I::from_usize(pi) {
                    Some(val) => val,
                    None => return index_out_of_range::<_, I>(pi),
                });
                Ok(())
            };

            for polygon in polygons {
                match polygon {
                    Polygon::P(ref vec) if vec.len() == 3 => {
                        for &pi in vec {
                            map(pi)?
                        }
                    }
                    Polygon::PT(ref vec) | Polygon::PN(ref vec) if vec.len() == 3 => {
                        for &(pi, _) in vec {
                            map(pi)?
                        }
                    }
                    Polygon::PTN(ref vec) if vec.len() == 3 => {
                        for &(pi, _, _) in vec {
                            map(pi)?
                        }
                    }
                    _ => make_error!(
                        UntriangulatedModel,
                        "Model should be triangulated first to be loaded properly"
                    ),
                }
            }
        }
        Ok((vb, ib))
    }
}

/// Vertex data type of `Obj` which contains position, normal and texture data of a vertex.
#[derive(Default, Copy, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "vulkano", repr(C))]
#[cfg_attr(feature = "vulkano", derive(Zeroable, Pod))]
pub struct TexturedVertex {
    /// Position vector of a vertex.
    pub position: [f32; 3],
    /// Normal vertor of a vertex.
    pub normal: [f32; 3],
    /// Texture of a vertex.
    pub texture: [f32; 3],
}

impl Point for TexturedVertex {
    fn position(&self) -> [f32; 3] {
        [self.position[0], self.position[1], self.position[2]]
    }
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct TexturedVertexIndex {
    pi: usize,
    ni: usize,
    ti: usize,
}

impl From<(usize, usize, usize)> for TexturedVertexIndex {
    fn from((pi, ti, ni): (usize, usize, usize)) -> Self {
        TexturedVertexIndex { pi, ni, ti }
    }
}

impl FromBufferIndexes<TexturedVertexIndex> for TexturedVertex {
    fn from_buffer_index(
        positions: &[(f32, f32, f32, f32)],
        normals: &[(f32, f32, f32)],
        tex_coords: &[(f32, f32, f32)],
        vertex_index: TexturedVertexIndex,
    ) -> Self {
        let position = positions[vertex_index.pi];
        let normal = normals[vertex_index.ni];
        let tex_coord = tex_coords[vertex_index.ti];
        TexturedVertex {
            position: [position.0, position.1, position.2],
            normal: [normal.0, normal.1, normal.2],
            texture: [tex_coord.0, tex_coord.1, tex_coord.2],
        }
    }
}

#[cfg(feature = "glium")]
implement_vertex!(TexturedVertex, position, normal, texture);
#[cfg(feature = "vulkano")]
impl_vertex!(TexturedVertex, position, normal, texture);

impl<I: ToPrimitive + FromPrimitive + Copy> FromRawVertex<I> for TexturedVertex {
    fn process(
        positions: Vec<(f32, f32, f32, f32)>,
        normals: Vec<(f32, f32, f32)>,
        tex_coords: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        let mut cache = VertexBufferCache::new(positions, normals, tex_coords, polygons.len());
        {
            for polygon in polygons {
                match polygon {
                    Polygon::P(_) => make_error!(InsufficientData, "Tried to extract normal and texture data which are not contained in the model"),
                    Polygon::PT(_) => make_error!(InsufficientData, "Tried to extract normal data which are not contained in the model"),
                    Polygon::PN(_) => make_error!(InsufficientData, "Tried to extract texture data which are not contained in the model"),
                    Polygon::PTN(vec) => {
                        cache.add_polygon(vec)?;
                    }
                    _ => make_error!(UntriangulatedModel, "Model should be triangulated first to be loaded properly")
                }
            }
        }
        Ok(cache.finalize())
    }
}

#[cfg(feature = "glium")]
mod glium_support {
    use super::Obj;
    use glium::backend::Facade;
    use glium::{index, vertex, IndexBuffer, VertexBuffer};

    impl<V: vertex::Vertex, I: glium::index::Index> Obj<V, I> {
        /// Retrieve glium-compatible vertex buffer from Obj
        pub fn vertex_buffer<F: Facade>(
            &self,
            facade: &F,
        ) -> Result<VertexBuffer<V>, vertex::BufferCreationError> {
            VertexBuffer::new(facade, &self.vertices)
        }

        /// Retrieve glium-compatible index buffer from Obj
        pub fn index_buffer<F: Facade>(
            &self,
            facade: &F,
        ) -> Result<IndexBuffer<I>, index::BufferCreationError> {
            IndexBuffer::new(facade, index::PrimitiveType::TrianglesList, &self.indices)
        }
    }
}

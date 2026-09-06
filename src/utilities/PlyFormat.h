#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// PLY container: vertices (position, normal, colour, UV, per-point attributes, arbitrary scalar
// fields), edges and triangle faces, read and written in either ASCII or binary_little_endian.
//
// Two ideas carry the whole file, and both replace something the original port duplicated:
//
//   1. VertexLayout is computed ONCE and consulted by both the header writer and the body writer.
//      Deciding independently in two places is how a file ends up announcing a property it never
//      writes -- which does not fail, it shifts every value after it.
//
//   2. PlyValueReader reads one typed scalar from either an ASCII line or a binary stream, so each
//      element is parsed ONCE instead of once per format. The two copies are otherwise identical
//      and drift apart the first time only one of them is fixed.
//
// Distinct from utilities/PlyMesh.h (triangle meshes) and utilities/PointCloudIO.h (ASCII points
// only). This is the general one; the other two predate it.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <Eigen/Core>
#include <Eigen/Geometry> // AlignedBox3f

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace util {

    using Vector3b = Eigen::Matrix<unsigned char, 3, 1>;

    // Per-point classification carried alongside the geometry.
    struct VoxelExtraAttrib {
        int label = 0;
        int deepLearningClass = 0;
    };

    // A coordinate only widens the bounding box if it is a real number. NaN would poison the box
    // permanently -- every later comparison against it is false, so the box would stop growing.
    inline bool IsUsableCoordinate(float value) { return std::isfinite(value); }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Point storage + bounding box, shared by every serializable format.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class HSerializable {
    public:
        virtual ~HSerializable() = default;

        virtual bool Serialize(const std::string &filename) = 0;
        virtual bool Deserialize(const std::string &filename) = 0;

        virtual void AddPoint(float x, float y, float z) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
            extendBounds(x, y, z);
        }

        // Four-component variant, for callers storing a homogeneous or weighted coordinate.
        // NOTE: PlyFormat's reader and writer both assume a stride of THREE, as does
        // GetPointCount(); a cloud built with this overload cannot be serialized.
        virtual void AddPoint(float x, float y, float z, float w) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
            points.push_back(w);
            extendBounds(x, y, z);
        }

        virtual void AddPointFloat3(const float *point) { AddPoint(point[0], point[1], point[2]); }

        virtual void AddPointFloat4(const float *point) {
            AddPoint(point[0], point[1], point[2], point[3]);
        }

        const std::vector<float> &GetPoints() const { return points; }
        std::vector<float> &GetPoints() { return points; }

        const Eigen::AlignedBox3f &GetAABB() const { return aabb; }

    protected:
        void extendBounds(float x, float y, float z) {
            if (IsUsableCoordinate(x) && IsUsableCoordinate(y) && IsUsableCoordinate(z))
                aabb.extend(Eigen::Vector3f(x, y, z));
        }

        std::vector<float> points;
        Eigen::AlignedBox3f aabb;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PlyFormat
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class PlyFormat : public HSerializable {
    public:
        enum class EDataType { Ascii, Binary };

        bool Serialize(const std::string &filename) override;
        bool Deserialize(const std::string &filename) override;

        std::size_t GetPointCount() const { return points.size() / 3; }

        const std::vector<float> &GetNormals() const { return normals; }
        const std::vector<float> &GetUVs() const { return uvs; }
        const std::vector<unsigned int> &GetEdgeIndices() const { return edgeIndices; }
        const std::vector<unsigned int> &GetIndices() const { return triangleIndices; }
        const std::vector<unsigned int> &GetTriangleIndices() const { return triangleIndices; }
        const std::vector<unsigned char> &GetColors() const { return colors; }
        std::vector<VoxelExtraAttrib> &GetExtraAttribs() { return extraAttribs; }
        const std::vector<VoxelExtraAttrib> &GetExtraAttribs() const { return extraAttribs; }
        bool UseAlpha() const { return useAlpha; }

        EDataType GetDataType() const { return dataType; }
        void SetDataType(EDataType type) { dataType = type; }

        // --- building -----------------------------------------------------------------------

        virtual void AddUV(float u, float v) {
            uvs.push_back(u);
            uvs.push_back(v);
        }
        virtual void AddUVFloat2(const float *uv) { AddUV(uv[0], uv[1]); }

        virtual void AddNormal(float x, float y, float z) {
            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);
        }
        virtual void AddNormalFloat3(const float *normal) {
            AddNormal(normal[0], normal[1], normal[2]);
        }

        virtual void AddEdgeIndex(unsigned int index) { edgeIndices.push_back(index); }
        virtual void AddTriangleIndex(unsigned int index) { triangleIndices.push_back(index); }

        virtual void AddFace(unsigned int i0, unsigned int i1, unsigned int i2) {
            triangleIndices.push_back(i0);
            triangleIndices.push_back(i1);
            triangleIndices.push_back(i2);
        }

        virtual void AddColor(unsigned char r, unsigned char g, unsigned char b) {
            colors.push_back(r);
            colors.push_back(g);
            colors.push_back(b);
        }

        virtual void AddColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
            colors.push_back(r);
            colors.push_back(g);
            colors.push_back(b);
            colors.push_back(a);
            useAlpha = true;
        }

        virtual void AddColor(float r, float g, float b) {
            AddColor(toByte(r), toByte(g), toByte(b));
        }

        virtual void AddColor(float r, float g, float b, float a) {
            AddColor(toByte(r), toByte(g), toByte(b), toByte(a));
        }

        virtual void AddColorFloat3(const float *color) { AddColor(color[0], color[1], color[2]); }
        virtual void AddColorFloat4(const float *color) {
            AddColor(color[0], color[1], color[2], color[3]);
        }

        virtual void AddColor(const Vector3b &color) { AddColor(color.x(), color.y(), color.z()); }
        virtual void AddColor(const Vector3b &color, unsigned char a) {
            AddColor(color.x(), color.y(), color.z(), a);
        }
        void AddColorUChar3(const unsigned char *color) {
            AddColor(color[0], color[1], color[2]);
        }

        virtual void SetColor(std::size_t index, float color) {
            if (index < colors.size()) colors[index] = toByte(color);
        }

        virtual void AddExtraAttrib(const VoxelExtraAttrib &extraAttrib) {
            extraAttribs.push_back(extraAttrib);
        }

        virtual void AddScalarField(const std::string &field, float value) {
            scalarFields[field].push_back(value);
        }

        bool HasScalarField(const std::string &field) const {
            return scalarFields.find(field) != scalarFields.end();
        }

        // Deserialize()가 PLY의 커스텀 float property(예: "class")를 읽어들이면 scalarFields에 채워진다.
        // field가 없으면 빈 벡터를 반환한다.
        const std::vector<float> &GetScalarField(const std::string &field) const {
            static const std::vector<float> empty;
            const auto it = scalarFields.find(field);
            return it != scalarFields.end() ? it->second : empty;
        }

    private:
        static unsigned char toByte(float unitValue) {
            const float scaled = unitValue * 255.0f;
            if (!(scaled > 0.0f)) return 0;             // also catches NaN
            return scaled >= 255.0f ? 255 : (unsigned char) (scaled);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Writing
        ///////////////////////////////////////////////////////////////////////////////////////////

        // What one vertex row actually contains. Decided once, then used to write BOTH the header
        // and the body, so the two cannot disagree about a property.
        struct VertexLayout {
            bool hasNormals = false;
            bool hasColors = false;
            bool hasAlpha = false;
            bool hasUVs = false;
            bool hasExtraAttribs = false;
            std::vector<std::string> scalarFieldNames; // only those with one value per point
        };

        VertexLayout describeVertices() const {
            const std::size_t pointCount = GetPointCount();
            VertexLayout layout;
            layout.hasNormals = normals.size() == points.size();
            layout.hasAlpha = useAlpha;
            // Colours are stored 3- or 4-wide depending on whether alpha was ever supplied, so the
            // expected length follows useAlpha rather than being tested both ways.
            layout.hasColors = pointCount > 0 && colors.size() == pointCount * (useAlpha ? 4u : 3u);
            layout.hasUVs = uvs.size() == pointCount * 2;
            layout.hasExtraAttribs = extraAttribs.size() == pointCount;
            for (const auto &field: scalarFields)
                if (field.second.size() == pointCount) layout.scalarFieldNames.push_back(field.first);
            return layout;
        }

        void writeHeader(std::ostream &stream, const VertexLayout &layout) const {
            stream << "ply\n";
            stream << (dataType == EDataType::Ascii ? "format ascii 1.0\n"
                                                    : "format binary_little_endian 1.0\n");

            stream << "element vertex " << GetPointCount() << "\n";
            stream << "property float x\nproperty float y\nproperty float z\n";
            if (layout.hasNormals) stream << "property float nx\nproperty float ny\nproperty float nz\n";
            if (layout.hasColors) {
                stream << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
                if (layout.hasAlpha) stream << "property uchar alpha\n";
            }
            if (layout.hasUVs) stream << "property float u\nproperty float v\n";
            if (layout.hasExtraAttribs)
                stream << "property int label\nproperty int deepLearningClass\n";
            for (const std::string &name: layout.scalarFieldNames)
                stream << "property float " << name << "\n";

            if (!edgeIndices.empty()) {
                stream << "element edge " << edgeIndices.size() / 2 << "\n";
                stream << "property int vertex1\nproperty int vertex2\n";
            }
            if (!triangleIndices.empty()) {
                stream << "element face " << triangleIndices.size() / 3 << "\n";
                stream << "property list uchar int vertex_indices\n";
            }
            stream << "end_header\n";
        }

        void writeAsciiBody(std::ostream &stream, const VertexLayout &layout) const {
            std::ostringstream row;
            row.precision(6);

            const std::size_t pointCount = GetPointCount();
            const unsigned colorStride = layout.hasAlpha ? 4u : 3u;
            for (std::size_t i = 0; i < pointCount; ++i) {
                row << points[3 * i] << ' ' << points[3 * i + 1] << ' ' << points[3 * i + 2];
                if (layout.hasNormals)
                    row << ' ' << normals[3 * i] << ' ' << normals[3 * i + 1] << ' ' << normals[3 * i + 2];
                if (layout.hasColors) {
                    row << ' ' << int(colors[colorStride * i]) << ' ' << int(colors[colorStride * i + 1])
                        << ' ' << int(colors[colorStride * i + 2]);
                    if (layout.hasAlpha) row << ' ' << int(colors[colorStride * i + 3]);
                }
                if (layout.hasUVs) row << ' ' << uvs[2 * i] << ' ' << uvs[2 * i + 1];
                if (layout.hasExtraAttribs)
                    row << ' ' << extraAttribs[i].label << ' ' << extraAttribs[i].deepLearningClass;
                for (const std::string &name: layout.scalarFieldNames)
                    row << ' ' << scalarFields.at(name)[i];
                row << '\n';

                // Flushed in batches rather than per point: one ostream insertion per value on a
                // multi-million-point cloud is dominated by stream overhead.
                if ((i + 1) % 10000 == 0) {
                    stream << row.str();
                    row.str(std::string());
                }
            }
            stream << row.str();

            for (std::size_t i = 0; i < edgeIndices.size() / 2; ++i)
                stream << edgeIndices[2 * i] << ' ' << edgeIndices[2 * i + 1] << '\n';
            for (std::size_t i = 0; i < triangleIndices.size() / 3; ++i)
                stream << "3 " << triangleIndices[3 * i] << ' ' << triangleIndices[3 * i + 1] << ' '
                       << triangleIndices[3 * i + 2] << '\n';
        }

        void writeBinaryBody(std::ostream &stream, const VertexLayout &layout) const {
            const auto writeFloat = [&stream](float value) {
                stream.write(reinterpret_cast<const char *>(&value), sizeof(float));
            };
            const auto writeInt = [&stream](std::int32_t value) {
                stream.write(reinterpret_cast<const char *>(&value), sizeof(std::int32_t));
            };
            const auto writeByte = [&stream](unsigned char value) {
                stream.write(reinterpret_cast<const char *>(&value), sizeof(unsigned char));
            };

            const std::size_t pointCount = GetPointCount();
            const unsigned colorStride = layout.hasAlpha ? 4u : 3u;
            for (std::size_t i = 0; i < pointCount; ++i) {
                writeFloat(points[3 * i]);
                writeFloat(points[3 * i + 1]);
                writeFloat(points[3 * i + 2]);
                if (layout.hasNormals) {
                    writeFloat(normals[3 * i]);
                    writeFloat(normals[3 * i + 1]);
                    writeFloat(normals[3 * i + 2]);
                }
                if (layout.hasColors) {
                    writeByte(colors[colorStride * i]);
                    writeByte(colors[colorStride * i + 1]);
                    writeByte(colors[colorStride * i + 2]);
                    if (layout.hasAlpha) writeByte(colors[colorStride * i + 3]);
                }
                if (layout.hasUVs) {
                    writeFloat(uvs[2 * i]);
                    writeFloat(uvs[2 * i + 1]);
                }
                if (layout.hasExtraAttribs) {
                    writeInt(extraAttribs[i].label);
                    writeInt(extraAttribs[i].deepLearningClass);
                }
                for (const std::string &name: layout.scalarFieldNames)
                    writeFloat(scalarFields.at(name)[i]);
            }

            for (std::size_t i = 0; i < edgeIndices.size() / 2; ++i) {
                writeInt(std::int32_t(edgeIndices[2 * i]));
                writeInt(std::int32_t(edgeIndices[2 * i + 1]));
            }
            for (std::size_t i = 0; i < triangleIndices.size() / 3; ++i) {
                writeByte(3);
                writeInt(std::int32_t(triangleIndices[3 * i]));
                writeInt(std::int32_t(triangleIndices[3 * i + 1]));
                writeInt(std::int32_t(triangleIndices[3 * i + 2]));
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Reading
        ///////////////////////////////////////////////////////////////////////////////////////////

        struct PropertyInfo {
            std::string type;
            std::string name;
            bool isList = false;
            std::string listCountType;
            std::string listContentType;
        };

        struct ElementInfo {
            std::string name;
            std::size_t count = 0;
            std::vector<PropertyInfo> properties;
        };

        // One typed scalar, from an ASCII line or from raw little-endian bytes. Every element below
        // is parsed against this, so it is written once rather than once per format.
        class PlyValueReader {
        public:
            PlyValueReader(std::istream &stream, bool binary) : m_stream(stream), m_binary(binary) {}

            // ASCII rows are line-delimited; binary rows are just the next bytes.
            bool BeginRow() {
                if (m_binary) return bool(m_stream);
                std::string line;
                if (!std::getline(m_stream, line)) return false;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                m_row.clear();
                m_row.str(line);
                return true;
            }

            float Scalar(const std::string &type) {
                if (!m_binary) {
                    float value = 0.0f;
                    m_row >> value; // "255" parses as a float just as well as "0.5"
                    return value;
                }
                if (type == "float" || type == "float32") return readRaw<float>();
                if (type == "double" || type == "float64") return float(readRaw<double>());
                if (type == "int" || type == "int32") return float(readRaw<std::int32_t>());
                if (type == "uint" || type == "uint32") return float(readRaw<std::uint32_t>());
                if (type == "short" || type == "int16") return float(readRaw<std::int16_t>());
                if (type == "ushort" || type == "uint16") return float(readRaw<std::uint16_t>());
                if (type == "char" || type == "int8") return float(readRaw<std::int8_t>());
                if (type == "uchar" || type == "uint8") return float(readRaw<std::uint8_t>());
                return 0.0f;
            }

            int Integer(const std::string &type) { return int(Scalar(type)); }

            static std::size_t BinaryWidth(const std::string &type) {
                if (type == "double" || type == "float64") return 8;
                if (type == "float" || type == "float32" || type == "int" || type == "int32" ||
                    type == "uint" || type == "uint32")
                    return 4;
                if (type == "short" || type == "int16" || type == "ushort" || type == "uint16") return 2;
                return 1;
            }

        private:
            template<typename T>
            T readRaw() {
                T value{};
                m_stream.read(reinterpret_cast<char *>(&value), sizeof(T));
                return value;
            }

            std::istream &m_stream;
            bool m_binary;
            std::istringstream m_row;
        };

        // Reads up to and including "end_header". Returns false if the stream is not a PLY.
        bool parseHeader(std::istream &stream, std::vector<ElementInfo> &elements, bool &binary) {
            std::string line;
            bool sawMagic = false;
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::istringstream tokens(line);
                std::string word;
                tokens >> word;

                if (word == "ply" || word == "PLY") {
                    sawMagic = true;
                } else if (word == "format") {
                    std::string format;
                    tokens >> format;
                    binary = format == "binary_little_endian";
                    SetDataType(binary ? EDataType::Binary : EDataType::Ascii);
                } else if (word == "element") {
                    ElementInfo element;
                    tokens >> element.name >> element.count;
                    elements.push_back(element);
                } else if (word == "property") {
                    if (elements.empty()) continue; // a property before any element: ignore it
                    PropertyInfo property;
                    tokens >> property.type;
                    if (property.type == "list") {
                        property.isList = true;
                        tokens >> property.listCountType >> property.listContentType >> property.name;
                    } else {
                        tokens >> property.name;
                        if (property.name == "a") property.name = "alpha";
                    }
                    elements.back().properties.push_back(property);
                } else if (word == "end_header") {
                    return sawMagic;
                }
            }
            return false; // ran out of input before end_header
        }

        // Any vertex property that is not one this class models becomes a scalar field.
        static bool isModelledVertexProperty(const std::string &name) {
            return name == "x" || name == "y" || name == "z" ||
                   name == "nx" || name == "ny" || name == "nz" ||
                   name == "red" || name == "green" || name == "blue" || name == "alpha" ||
                   name == "u" || name == "v" ||
                   name == "label" || name == "deepLearningClass";
        }

        void readVertices(const ElementInfo &element, PlyValueReader &reader) {
            points.reserve(points.size() + element.count * 3);

            for (std::size_t i = 0; i < element.count; ++i) {
                if (!reader.BeginRow()) return;

                float x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0, u = 0, v = 0;
                unsigned char r = 255, g = 255, b = 255, a = 255;
                int label = 0, deepLearningClass = 0;
                bool hasNormal = false, hasColor = false, hasUV = false, hasExtra = false;

                for (const PropertyInfo &property: element.properties) {
                    const float value = reader.Scalar(property.type);
                    const std::string &name = property.name;

                    if (name == "x") x = value;
                    else if (name == "y") y = value;
                    else if (name == "z") z = value;
                    else if (name == "nx") { nx = value; hasNormal = true; }
                    else if (name == "ny") { ny = value; hasNormal = true; }
                    else if (name == "nz") { nz = value; hasNormal = true; }
                    else if (name == "red") { r = (unsigned char) value; hasColor = true; }
                    else if (name == "green") { g = (unsigned char) value; hasColor = true; }
                    else if (name == "blue") { b = (unsigned char) value; hasColor = true; }
                    else if (name == "alpha") { a = (unsigned char) value; hasColor = true; useAlpha = true; }
                    else if (name == "u") { u = value; hasUV = true; }
                    else if (name == "v") { v = value; hasUV = true; }
                    else if (name == "label") { label = int(value); hasExtra = true; }
                    else if (name == "deepLearningClass") { deepLearningClass = int(value); hasExtra = true; }
                    else scalarFields[name].push_back(value);
                }

                AddPoint(x, y, z);
                if (hasNormal) AddNormal(nx, ny, nz);
                if (hasColor) {
                    if (useAlpha) AddColor(r, g, b, a);
                    else AddColor(r, g, b);
                }
                if (hasUV) AddUV(u, v);
                if (hasExtra) AddExtraAttrib(VoxelExtraAttrib{label, deepLearningClass});
            }
        }

        void readFaces(const ElementInfo &element, PlyValueReader &reader) {
            triangleIndices.reserve(triangleIndices.size() + element.count * 3);

            for (std::size_t i = 0; i < element.count; ++i) {
                if (!reader.BeginRow()) return;

                for (const PropertyInfo &property: element.properties) {
                    if (!property.isList) {
                        reader.Scalar(property.type); // a per-face scalar this class does not keep
                        continue;
                    }
                    const int cornerCount = reader.Integer(property.listCountType);
                    std::vector<int> corners;
                    corners.reserve(std::size_t(cornerCount < 0 ? 0 : cornerCount));
                    for (int k = 0; k < cornerCount; ++k)
                        corners.push_back(reader.Integer(property.listContentType));

                    // Fan-triangulated around corner 0, for a polygon of ANY size. The original
                    // handled only triangles and quads, which drops every larger face on the floor
                    // -- a mesh exported with n-gons then loads with holes and no error.
                    for (std::size_t k = 1; k + 1 < corners.size(); ++k)
                        AddFace((unsigned int) corners[0], (unsigned int) corners[k],
                                (unsigned int) corners[k + 1]);
                }
            }
        }

        void readEdges(const ElementInfo &element, PlyValueReader &reader) {
            for (std::size_t i = 0; i < element.count; ++i) {
                if (!reader.BeginRow()) return;

                int first = -1, second = -1;
                for (const PropertyInfo &property: element.properties) {
                    const int value = reader.Integer(property.type);
                    if (property.name == "vertex1") first = value;
                    else if (property.name == "vertex2") second = value;
                }
                if (first >= 0 && second >= 0) {
                    edgeIndices.push_back((unsigned int) first);
                    edgeIndices.push_back((unsigned int) second);
                }
            }
        }

        // An element this class does not model still has to be stepped over exactly, or every
        // element after it is read from the wrong offset.
        bool skipElement(const ElementInfo &element, std::istream &stream, PlyValueReader &reader,
                         bool binary) {
            if (!binary) {
                std::string line;
                for (std::size_t i = 0; i < element.count; ++i)
                    if (!std::getline(stream, line)) return false;
                return true;
            }

            std::size_t bytesPerElement = 0;
            for (const PropertyInfo &property: element.properties) {
                // A list has a per-row length, so the element's size is not known in advance and
                // seeking past it is impossible. Refuse rather than resume at a wrong offset.
                if (property.isList) return false;
                bytesPerElement += PlyValueReader::BinaryWidth(property.type);
            }
            (void) reader;
            stream.seekg(std::streamoff(bytesPerElement * element.count), std::ios::cur);
            return bool(stream);
        }

        EDataType dataType = EDataType::Binary;

        std::vector<float> uvs;
        std::vector<float> normals;
        std::vector<unsigned int> edgeIndices;
        std::vector<unsigned int> triangleIndices;
        std::vector<unsigned char> colors;
        std::vector<VoxelExtraAttrib> extraAttribs;
        bool useAlpha = false;

        std::unordered_map<std::string, std::vector<float>> scalarFields;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////

    inline bool PlyFormat::Serialize(const std::string &filename) {
        std::ofstream stream(filename, std::ios::out | std::ios::binary);
        if (!stream.is_open()) return false;

        const VertexLayout layout = describeVertices();
        writeHeader(stream, layout);
        if (dataType == EDataType::Ascii) writeAsciiBody(stream, layout);
        else writeBinaryBody(stream, layout);

        return bool(stream);
    }

    inline bool PlyFormat::Deserialize(const std::string &filename) {
        std::ifstream stream(filename, std::ios::in | std::ios::binary);
        if (!stream.is_open()) return false;

        std::vector<ElementInfo> elements;
        bool binary = false;
        if (!parseHeader(stream, elements, binary)) return false;

        // Declared before the first row is read so a field present in the header but absent from
        // some row still exists (empty) rather than being missing entirely.
        for (const ElementInfo &element: elements) {
            if (element.name != "vertex") continue;
            for (const PropertyInfo &property: element.properties)
                if (!property.isList && !isModelledVertexProperty(property.name))
                    scalarFields[property.name].reserve(element.count);
        }

        PlyValueReader reader(stream, binary);
        for (const ElementInfo &element: elements) {
            if (element.name == "vertex") readVertices(element, reader);
            else if (element.name == "face") readFaces(element, reader);
            else if (element.name == "edge") readEdges(element, reader);
            else if (!skipElement(element, stream, reader, binary)) return false;
        }
        return true;
    }

} // namespace util

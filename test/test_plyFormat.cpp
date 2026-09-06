#include "utilities/PlyFormat.h"
#include "utilities/PlyMesh.h"
#include "utilities/PointCloudIO.h" // the legacy entry points, now adapters

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace {

    namespace fs = std::filesystem;

    struct TempPly {
        fs::path path;
        explicit TempPly(const std::string &name)
            : path(fs::temp_directory_path() / ("vkbvh_ply_" + name + ".ply")) {
            fs::remove(path);
        }
        ~TempPly() { fs::remove(path); }
        std::string String() const { return path.string(); }
    };

    // Positions that are exact in float, so a round trip is bit-for-bit and a failure means the
    // format lost data rather than that the comparison was too tight.
    util::PlyFormat makeCloud(bool withNormals, bool withColors, bool withAlpha) {
        util::PlyFormat ply;
        for (int i = 0; i < 7; ++i) {
            ply.AddPoint(float(i) * 0.25f, float(i) * -0.5f, float(i) + 0.125f);
            if (withNormals) ply.AddNormal(0.0f, 0.0f, 1.0f);
            if (withColors) {
                if (withAlpha)
                    ply.AddColor((unsigned char) (i * 10), (unsigned char) (i * 20),
                                 (unsigned char) (i * 30), (unsigned char) (i * 5));
                else
                    ply.AddColor((unsigned char) (i * 10), (unsigned char) (i * 20),
                                 (unsigned char) (i * 30));
            }
        }
        return ply;
    }

} // namespace

// The core contract: what went in comes back, in both encodings. Asserted per component rather
// than on counts -- a wrong stride or a mis-sized binary field still yields the right NUMBER of
// points, and every value after it is shifted.
TEST(PlyFormat, RoundTripsPointsAndNormalsInBothEncodings) {
    for (const util::PlyFormat::EDataType encoding:
         {util::PlyFormat::EDataType::Ascii, util::PlyFormat::EDataType::Binary}) {
        const bool ascii = encoding == util::PlyFormat::EDataType::Ascii;
        TempPly file(ascii ? "roundtrip_ascii" : "roundtrip_binary");

        util::PlyFormat written = makeCloud(/*withNormals=*/true, /*withColors=*/false, false);
        written.SetDataType(encoding);
        ASSERT_TRUE(written.Serialize(file.String())) << (ascii ? "ascii" : "binary");

        util::PlyFormat read;
        ASSERT_TRUE(read.Deserialize(file.String())) << (ascii ? "ascii" : "binary");

        ASSERT_EQ(read.GetPointCount(), written.GetPointCount());
        ASSERT_EQ(read.GetNormals().size(), written.GetNormals().size());
        for (std::size_t i = 0; i < written.GetPoints().size(); ++i)
            EXPECT_FLOAT_EQ(read.GetPoints()[i], written.GetPoints()[i]) << "point value " << i;
        for (std::size_t i = 0; i < written.GetNormals().size(); ++i)
            EXPECT_FLOAT_EQ(read.GetNormals()[i], written.GetNormals()[i]) << "normal value " << i;
        EXPECT_EQ(read.GetDataType(), encoding) << "the reader did not report the encoding it found";
    }
}

// Colour is stored 3- or 4-wide depending on whether alpha was ever supplied. The header and the
// body used to decide that independently, which produced a file announcing RGB and containing
// RGBA (or nothing) -- readable, and wrong from the first colour onward.
TEST(PlyFormat, RoundTripsColorsWithAndWithoutAlpha) {
    for (const bool withAlpha: {false, true}) {
        for (const util::PlyFormat::EDataType encoding:
             {util::PlyFormat::EDataType::Ascii, util::PlyFormat::EDataType::Binary}) {
            TempPly file(std::string("color_") + (withAlpha ? "rgba_" : "rgb_") +
                         (encoding == util::PlyFormat::EDataType::Ascii ? "ascii" : "binary"));

            util::PlyFormat written = makeCloud(/*withNormals=*/false, /*withColors=*/true, withAlpha);
            written.SetDataType(encoding);
            ASSERT_TRUE(written.Serialize(file.String()));

            util::PlyFormat read;
            ASSERT_TRUE(read.Deserialize(file.String()));

            EXPECT_EQ(read.UseAlpha(), withAlpha);
            ASSERT_EQ(read.GetColors().size(), written.GetColors().size())
                    << "alpha=" << withAlpha;
            for (std::size_t i = 0; i < written.GetColors().size(); ++i)
                EXPECT_EQ(read.GetColors()[i], written.GetColors()[i]) << "colour byte " << i;
        }
    }
}

// UVs are two floats per point. The original wrote them at a stride of three and gated them on
// `uvs.size() == points.size()`, which is only ever true for an empty cloud -- so UVs were
// silently never written. This is the test that would have caught that.
TEST(PlyFormat, RoundTripsUVs) {
    TempPly file("uv");
    util::PlyFormat written;
    for (int i = 0; i < 5; ++i) {
        written.AddPoint(float(i), 0.0f, 0.0f);
        written.AddUV(float(i) * 0.25f, float(i) * 0.5f);
    }
    written.SetDataType(util::PlyFormat::EDataType::Binary);
    ASSERT_TRUE(written.Serialize(file.String()));

    util::PlyFormat read;
    ASSERT_TRUE(read.Deserialize(file.String()));
    ASSERT_EQ(read.GetUVs().size(), written.GetUVs().size()) << "UVs were not written at all";
    for (std::size_t i = 0; i < written.GetUVs().size(); ++i)
        EXPECT_FLOAT_EQ(read.GetUVs()[i], written.GetUVs()[i]) << "uv value " << i;
}

TEST(PlyFormat, RoundTripsFacesAndEdges) {
    TempPly file("topology");
    util::PlyFormat written;
    for (int i = 0; i < 4; ++i) written.AddPoint(float(i), float(i), 0.0f);
    written.AddFace(0, 1, 2);
    written.AddFace(1, 2, 3);
    written.AddEdgeIndex(0);
    written.AddEdgeIndex(3);
    written.SetDataType(util::PlyFormat::EDataType::Binary);
    ASSERT_TRUE(written.Serialize(file.String()));

    util::PlyFormat read;
    ASSERT_TRUE(read.Deserialize(file.String()));
    EXPECT_EQ(read.GetTriangleIndices(), written.GetTriangleIndices());
    EXPECT_EQ(read.GetEdgeIndices(), written.GetEdgeIndices());
}

// A scalar field is only written when it has one value per point. Announcing one in the header
// and then skipping it in the body shifts every value after it on every row.
TEST(PlyFormat, RoundTripsScalarFieldsAndDropsRaggedOnes) {
    TempPly file("scalar");
    util::PlyFormat written;
    for (int i = 0; i < 6; ++i) {
        written.AddPoint(float(i), 0.0f, 0.0f);
        written.AddScalarField("curvature", float(i) * 0.5f);
    }
    written.AddScalarField("ragged", 1.0f); // one value for six points: must not be written
    written.SetDataType(util::PlyFormat::EDataType::Ascii);
    ASSERT_TRUE(written.Serialize(file.String()));

    util::PlyFormat read;
    ASSERT_TRUE(read.Deserialize(file.String()));
    ASSERT_TRUE(read.HasScalarField("curvature"));
    ASSERT_EQ(read.GetScalarField("curvature").size(), std::size_t(6));
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(read.GetScalarField("curvature")[std::size_t(i)], float(i) * 0.5f);
    EXPECT_FALSE(read.HasScalarField("ragged"))
            << "a field with fewer values than points was announced in the header";
    EXPECT_EQ(read.GetPointCount(), std::size_t(6)) << "the row layout drifted";
}

// The repo already has an ASCII PLY reader that four tools depend on. If the two disagree about
// the same file, one of them is wrong -- and nothing else in the suite would notice.
TEST(PlyFormat, AsciiOutputIsReadableByTheExistingLoader) {
    TempPly file("interop");
    util::PlyFormat written = makeCloud(/*withNormals=*/true, /*withColors=*/false, false);
    written.SetDataType(util::PlyFormat::EDataType::Ascii);
    ASSERT_TRUE(written.Serialize(file.String()));

    std::vector<Eigen::Vector3f> points, normals;
    ASSERT_TRUE(util::LoadPly(file.String(), points, normals));
    ASSERT_EQ(points.size(), written.GetPointCount());
    ASSERT_EQ(normals.size(), written.GetPointCount());
    for (std::size_t i = 0; i < points.size(); ++i) {
        EXPECT_FLOAT_EQ(points[i].x(), written.GetPoints()[3 * i]);
        EXPECT_FLOAT_EQ(points[i].y(), written.GetPoints()[3 * i + 1]);
        EXPECT_FLOAT_EQ(points[i].z(), written.GetPoints()[3 * i + 2]);
    }
}

// The bounding box must ignore a non-finite coordinate. Extending by NaN makes every later
// comparison false, so the box silently stops growing for the rest of the cloud.
TEST(PlyFormat, BoundingBoxIgnoresNonFiniteCoordinates) {
    util::PlyFormat ply;
    ply.AddPoint(1.0f, 2.0f, 3.0f);
    ply.AddPoint(std::nanf(""), 0.0f, 0.0f);
    ply.AddPoint(-1.0f, -2.0f, -3.0f);

    EXPECT_EQ(ply.GetPointCount(), std::size_t(3)) << "the point itself is still stored";
    EXPECT_FLOAT_EQ(ply.GetAABB().min().x(), -1.0f);
    EXPECT_FLOAT_EQ(ply.GetAABB().max().z(), 3.0f);
}

// An element this class does not model has to be stepped over exactly, or every element after it
// is read from the wrong offset. Asserted through the faces that FOLLOW the unknown element.
TEST(PlyFormat, SkipsAnUnknownAsciiElementWithoutLosingAlignment) {
    TempPly file("unknown_element");
    {
        std::ofstream out(file.String());
        out << "ply\nformat ascii 1.0\n"
            << "element vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
            << "element material 2\nproperty float shininess\n"
            << "element face 1\nproperty list uchar int vertex_indices\n"
            << "end_header\n"
            << "0 0 0\n1 1 1\n"
            << "0.5\n0.75\n"
            << "3 0 1 0\n";
    }

    util::PlyFormat read;
    ASSERT_TRUE(read.Deserialize(file.String()));
    EXPECT_EQ(read.GetPointCount(), std::size_t(2));
    ASSERT_EQ(read.GetTriangleIndices().size(), std::size_t(3))
            << "the unknown element was not skipped by exactly its row count";
    EXPECT_EQ(read.GetTriangleIndices()[1], 1u);
}

// The point of routing util::LoadPly through PlyFormat: the hand-rolled reader it replaced
// returned false on anything but ASCII, so every tool built on it silently could not open a
// binary PLY. Four example tools and FrameLoader go through this entry point.
TEST(PlyFormat, TheLegacyLoaderNowReadsBinaryFiles) {
    TempPly file("legacy_reads_binary");
    util::PlyFormat written = makeCloud(/*withNormals=*/true, /*withColors=*/false, false);
    written.SetDataType(util::PlyFormat::EDataType::Binary);
    ASSERT_TRUE(written.Serialize(file.String()));

    std::vector<Eigen::Vector3f> points, normals;
    ASSERT_TRUE(util::LoadPly(file.String(), points, normals))
            << "util::LoadPly still refuses binary_little_endian";
    ASSERT_EQ(points.size(), written.GetPointCount());
    ASSERT_EQ(normals.size(), written.GetPointCount());
    for (std::size_t i = 0; i < points.size(); ++i)
        EXPECT_FLOAT_EQ(points[i].y(), written.GetPoints()[3 * i + 1]) << "point " << i;
}

// util::LoadPly APPENDS, so a caller can accumulate several files into one pair of vectors.
// Common::PointCloud::Load replaces; the two must not be confused.
TEST(PlyFormat, TheLegacyLoaderStillAppends) {
    TempPly file("legacy_appends");
    util::PlyFormat written = makeCloud(/*withNormals=*/true, /*withColors=*/false, false);
    written.SetDataType(util::PlyFormat::EDataType::Ascii);
    ASSERT_TRUE(written.Serialize(file.String()));

    std::vector<Eigen::Vector3f> points, normals;
    ASSERT_TRUE(util::LoadPly(file.String(), points, normals));
    ASSERT_TRUE(util::LoadPly(file.String(), points, normals));
    EXPECT_EQ(points.size(), written.GetPointCount() * 2);
    EXPECT_EQ(normals.size(), written.GetPointCount() * 2);
}

// PlyMesh fan-triangulates polygons of any size. PlyFormat used to handle only triangles and
// quads, so routing the mesh loader through it would have dropped every larger face silently --
// which is exactly the Artec-style export PlyMesh was written for.
TEST(PlyFormat, FanTriangulatesPolygonsLargerThanAQuad) {
    TempPly file("ngon");
    {
        std::ofstream out(file.String());
        out << "ply\nformat ascii 1.0\n"
            << "element vertex 5\nproperty float x\nproperty float y\nproperty float z\n"
            << "element face 1\nproperty list uchar int vertex_indices\n"
            << "end_header\n"
            << "0 0 0\n1 0 0\n2 1 0\n1 2 0\n0 1 0\n"
            << "5 0 1 2 3 4\n";
    }

    util::TriMesh mesh;
    ASSERT_NO_THROW(util::LoadPlyMesh(file.String(), mesh));
    EXPECT_EQ(mesh.vertices.size(), std::size_t(5));
    // A 5-gon fans into 3 triangles; the old 3-or-4 special case produced none.
    ASSERT_EQ(mesh.faces.size(), std::size_t(3));
    EXPECT_EQ(mesh.faces[0], Eigen::Vector3i(0, 1, 2));
    EXPECT_EQ(mesh.faces[2], Eigen::Vector3i(0, 3, 4));
}

TEST(PlyFormat, MeshLoaderReadsBinaryFaces) {
    TempPly file("mesh_binary");
    util::PlyFormat written;
    for (int i = 0; i < 4; ++i) written.AddPoint(float(i), float(i % 2), 0.0f);
    written.AddFace(0, 1, 2);
    written.AddFace(1, 2, 3);
    written.SetDataType(util::PlyFormat::EDataType::Binary);
    ASSERT_TRUE(written.Serialize(file.String()));

    util::TriMesh mesh;
    ASSERT_NO_THROW(util::LoadPlyMesh(file.String(), mesh));
    EXPECT_EQ(mesh.vertices.size(), std::size_t(4));
    ASSERT_EQ(mesh.faces.size(), std::size_t(2));
    EXPECT_EQ(mesh.faces[1], Eigen::Vector3i(1, 2, 3));
}

// Against a real repo asset, not a fixture: the migration swapped the parser under four tools and
// FrameLoader, and a fixture that only round-trips this file's own writer cannot catch a reader
// that disagrees with the PLYs actually on disk.
TEST(PlyFormat, ReadsARepositoryAssetCompletely) {
    const fs::path asset = fs::path(VKBVH_SOURCE_DIR) / "scan_out" / "frame_0000.ply";
    if (!fs::exists(asset)) GTEST_SKIP() << "scan_out/frame_0000.ply not present";

    std::vector<Eigen::Vector3f> points, normals;
    ASSERT_TRUE(util::LoadPly(asset.string(), points, normals));
    // The count the file's own header declares.
    EXPECT_EQ(points.size(), std::size_t(37033));
    EXPECT_EQ(normals.size(), points.size()) << "this asset carries a normal per point";

    // Every coordinate finite: a stride slip turns normals into positions and shows up here.
    for (const Eigen::Vector3f &p: points)
        ASSERT_TRUE(p.allFinite());
    for (const Eigen::Vector3f &n: normals)
        ASSERT_NEAR(n.norm(), 1.0f, 1e-3f) << "normals are not unit length: fields are misaligned";
}

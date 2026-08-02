#include "utilities/PointCloudIO.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Eigen::Vector3f;
namespace fs = std::filesystem;

namespace {

    // A unique scratch path under the OS temp dir (removed by TempFile's dtor). Prefixed with the
    // running test's name so concurrent tests never share a file.
    struct TempFile {
        fs::path path;
        explicit TempFile(const std::string &name)
            : path(fs::temp_directory_path() /
                   (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
                    "_" + name)) {}
        ~TempFile() {
            std::error_code ec;
            fs::remove(path, ec);
        }
        std::string str() const { return path.string(); }
    };

    // Values chosen to survive the writer's default 6-significant-digit float formatting exactly.
    void sampleCloud(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm) {
        pts = {{1.5f, -2.25f, 3.125f}, {0.5f, 0.75f, -1.0f}, {10.0f, 20.0f, 30.0f}};
        nrm = {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.6f, 0.8f, 0.0f}};
    }

    void expectClouds(const std::vector<Vector3f> &a, const std::vector<Vector3f> &b) {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i].x(), b[i].x(), 1e-4f);
            EXPECT_NEAR(a[i].y(), b[i].y(), 1e-4f);
            EXPECT_NEAR(a[i].z(), b[i].z(), 1e-4f);
        }
    }

} // namespace

// PLY with normals: save -> load reproduces both points and normals.
TEST(PointCloudIO, PlyRoundtripWithNormals) {
    std::vector<Vector3f> pts, nrm;
    sampleCloud(pts, nrm);
    TempFile f("pc.ply");
    ASSERT_TRUE(util::SavePly(f.str(), pts, nrm));

    std::vector<Vector3f> rp, rn;
    ASSERT_TRUE(util::LoadPly(f.str(), rp, rn));
    expectClouds(pts, rp);
    expectClouds(nrm, rn);
}

// PLY without normals: writer omits nx/ny/nz, reader returns points only, no normals.
TEST(PointCloudIO, PlyPointsOnly) {
    std::vector<Vector3f> pts, nrm;
    sampleCloud(pts, nrm);
    nrm.clear();
    TempFile f("pc_nonrm.ply");
    ASSERT_TRUE(util::SavePly(f.str(), pts, nrm));

    std::vector<Vector3f> rp, rn;
    ASSERT_TRUE(util::LoadPly(f.str(), rp, rn));
    expectClouds(pts, rp);
    EXPECT_TRUE(rn.empty());
}

// LoadPly appends to existing vectors (callers accumulate frames into one cloud).
TEST(PointCloudIO, LoadPlyAppends) {
    std::vector<Vector3f> pts, nrm;
    sampleCloud(pts, nrm);
    TempFile f("pc_app.ply");
    ASSERT_TRUE(util::SavePly(f.str(), pts, nrm));

    std::vector<Vector3f> rp = {{99.0f, 99.0f, 99.0f}}, rn = {{0.0f, 1.0f, 0.0f}};
    ASSERT_TRUE(util::LoadPly(f.str(), rp, rn));
    ASSERT_EQ(rp.size(), pts.size() + 1);
    EXPECT_NEAR(rp[0].x(), 99.0f, 1e-4f);  // pre-existing entry kept
    EXPECT_NEAR(rp[1].x(), pts[0].x(), 1e-4f); // appended after it
}

// OBJ (as point cloud): v -> points, vn -> normals paired 1:1.
TEST(PointCloudIO, ObjRoundtrip) {
    std::vector<Vector3f> pts, nrm;
    sampleCloud(pts, nrm);
    TempFile f("pc.obj");
    ASSERT_TRUE(util::SaveObj(f.str(), pts, nrm));

    std::vector<Vector3f> rp, rn;
    ASSERT_TRUE(util::LoadObj(f.str(), rp, rn));
    expectClouds(pts, rp);
    expectClouds(nrm, rn);
}

// Extension dispatch: .obj routes to OBJ, everything else to PLY; each reads back its own format.
TEST(PointCloudIO, ExtensionDispatch) {
    std::vector<Vector3f> pts, nrm;
    sampleCloud(pts, nrm);

    TempFile obj("d.obj"), ply("d.ply");
    ASSERT_TRUE(util::SavePointCloud(obj.str(), pts, nrm));
    ASSERT_TRUE(util::SavePointCloud(ply.str(), pts, nrm));

    // The .obj file is OBJ (starts with "v "), the .ply file is PLY (starts with "ply").
    std::ifstream fo(obj.str()), fp(ply.str());
    std::string to, tp;
    fo >> to;
    fp >> tp;
    EXPECT_EQ(to, "v");
    EXPECT_EQ(tp, "ply");

    std::vector<Vector3f> rp, rn;
    ASSERT_TRUE(util::LoadPointCloud(obj.str(), rp, rn));
    expectClouds(pts, rp);
    expectClouds(nrm, rn);
}

// Missing / unreadable file returns false, leaves outputs empty.
TEST(PointCloudIO, LoadMissingReturnsFalse) {
    std::vector<Vector3f> rp, rn;
    EXPECT_FALSE(util::LoadPly((fs::temp_directory_path() / "no_such_file_xyz.ply").string(), rp, rn));
    EXPECT_TRUE(rp.empty());
}

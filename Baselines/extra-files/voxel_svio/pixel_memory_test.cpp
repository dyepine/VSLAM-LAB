#include "pixelSelector.h"
#include <cmath>
#include <stdexcept>
#include <vector>

// Exercise both cameras and the partial histogram blocks using the real frame
// pyramid and pixel selector. Under Memcheck this also detects undefined reads
// inside the selector, which a check of the returned coordinates cannot see.
static void check_image(int width, int height) {
    pyr_levels_used = 3;
    for (int level = 0; level < pyr_levels_used; ++level) {
        wG[level] = width >> level;
        hG[level] = height >> level;
    }
    std::vector<unsigned char> image(width * height), mask(width * height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image[x + y * width] = (x * 37 + y * 71 + (x * y) % 251) % 256;
    auto f = std::make_shared<frame>();
    f->makeImageLeft(image.data(), mask.data(), nullptr, 0.0);
    f->makeImageRight(image.data(), mask.data(), nullptr, 0.0);
    for (int side = 0; side < 2; ++side) {
        auto gradients = side ? f->abs_squared_grad_right : f->abs_squared_grad_left;
        auto pyramid = side ? f->dIp_right : f->dIp_left;
        for (int level = 0; level < pyr_levels_used; ++level) {
            const int w = wG[level], h = hG[level];
            for (int i = 0; i < w * h; ++i)
                if (!std::isfinite(gradients[level][i]) || gradients[level][i] < 0)
                    throw std::runtime_error("Undefined/non-finite gradient in pyramid");
            for (int x = 0; x < w; ++x)
                for (int y : {0, h - 1}) {
                    const int i = x + y * w;
                    if (gradients[level][i] != 0 ||
                        !pyramid[level][i].tail<2>().isZero(0))
                        throw std::runtime_error("Pyramid boundary gradient must be zero");
                }
        }
    }
    pixelSelector selector(width, height);
    const int counts[] = {selector.pixelSelectionLeft(f, 200),
                          selector.pixelSelectionRight(f, 200)};
    for (int side = 0; side < 2; ++side) {
        const float* selected = side ? f->selected_pixels_right : f->selected_pixels_left;
        int count = 0;
        for (int i = 0; i < width * height; ++i) {
            if (selected[i] < PIXSEL_VOID || selected[i] > PIXSEL_3 ||
                !std::isfinite(selected[i]))
                throw std::runtime_error("Invalid selected-pixel status");
            count += selected[i] != PIXSEL_VOID;
        }
        if (count == 0 || count != counts[side])
            throw std::runtime_error("Pixel selection lost texture or returned an incorrect count");
    }
    f->release();
}

int main() {
    cv::setNumThreads(1);
    check_image(128, 96);    // Complete histogram blocks.
    check_image(132, 100);   // Partial blocks on both axes.
    check_image(129, 97);    // Last blocks contain only excluded border pixels.
    check_image(752, 480);   // Author EuRoC image dimensions.
    check_image(1224, 1024); // GEODE dimensions that exposed the defect.
}

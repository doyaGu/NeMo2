#include <gtest/gtest.h>

#include "Player.h"

class PlayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        player_ = new Player();
    }

    void TearDown() override {
        delete player_;
    }

    Player *player_ = nullptr;
};

TEST_F(PlayerTest, Init) {
    ASSERT_NE(nullptr, player_);

    ASSERT_TRUE(player_->Init());
}

TEST_F(PlayerTest, Load) {
    ASSERT_NE(nullptr, player_);

    ASSERT_TRUE(player_->Load("base.cmo"));

    player_->Play();
    player_->Run();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
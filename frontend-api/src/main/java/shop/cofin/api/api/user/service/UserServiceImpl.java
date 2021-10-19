package shop.cofin.api.api.user.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import shop.cofin.api.api.user.domain.UserSerializer;
import shop.cofin.api.api.user.repository.UserRepository;

import java.util.Optional;

@Service @RequiredArgsConstructor
public class UserServiceImpl implements UserService{

    private final UserRepository userRepository;

    @Override
    public Optional<UserSerializer> findById(long userId) {
        return userRepository.findById(userId);
    }
}

package shop.cofin.api.api.user.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.user.domain.UserDTO;

import javax.transaction.Transactional;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    @Transactional
    @Query(value = "select u.username, " +
            "u.user_id userId, " +
            "u.name, " +
            "u.email, " +
            "u.reg_date regDate, " +
            "u.password  " +
            "from users u where u.username = :username and u.password = :password",
            nativeQuery = true)
//    @Query(value = "select * from users where users.username=:username " +
//            "and users.password=:password", nativeQuery = true)
    Optional<User> login(@Param("username") String username,
                         @Param("password") String password);
}

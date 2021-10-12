package shop.cofin.api.api.user.domain;


import com.sun.istack.NotNull;
import lombok.Data;
import org.springframework.stereotype.Component;
import shop.cofin.api.api.board.domain.Article;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Data
@Component
@Entity
@Table(name="users")
public class User {
    @Id
    @Column(name="user_id")
    @GeneratedValue    private long userId;
    @Column    private  @NotNull String username;
    @Column    private  @NotNull String password;
    @Column    private  @NotNull String name;
    @Column    private  @NotNull String email;
    @Column(name= "reg_date")    private  @NotNull String regDate;
    @OneToMany(mappedBy = "user")
    private List<Article> articleList= new ArrayList<>();

}

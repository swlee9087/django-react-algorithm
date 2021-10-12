package shop.cofin.api.api.board.domain;

import com.sun.istack.NotNull;
import lombok.Data;
import org.springframework.stereotype.Component;
import shop.cofin.api.api.user.domain.UserSerializer;

import javax.persistence.*;

@Data @Component @Entity @Table(name="articles")

public class Article {
    @Id
    @Column(name="article_id")
    @GeneratedValue
    private long articleId;
    @Column @NotNull     private String title;
    @Column @NotNull     private String content;
    @Column(name="written_date") @NotNull     private String writtenDate;

    @ManyToOne(fetch = FetchType.LAZY)    @JoinColumn(name="user_id") private UserSerializer user;
    @ManyToOne(fetch = FetchType.LAZY)    @JoinColumn(name="item_id") private UserSerializer item;


}
